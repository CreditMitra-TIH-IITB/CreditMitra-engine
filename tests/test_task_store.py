"""Task results at rest: opt-in encryption, retention, and task ids that cannot
escape the tasks directory."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from app.core.config import settings
from app.services import task_store

SECRET = "RAMESH KUMAR SALARY"


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "tasks").mkdir()
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "TASK_ENCRYPTION_KEY", "")
    monkeypatch.setattr(settings, "RESULT_RETENTION_DAYS", 0)
    return tmp_path


def _task_id() -> str:
    return str(uuid.uuid4())


def test_without_a_key_results_are_plain_json_as_before(data_dir: Path) -> None:
    tid = _task_id()
    task_store.update_task_status(tid, "completed", note=SECRET)
    stored = data_dir / "tasks" / f"{tid}.json"
    assert json.loads(stored.read_text(encoding="utf-8"))["note"] == SECRET


def test_with_a_key_results_are_encrypted_and_readable(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "TASK_ENCRYPTION_KEY", Fernet.generate_key().decode())
    tid = _task_id()
    task_store.update_task_status(tid, "completed", note=SECRET)

    stored = data_dir / "tasks" / f"{tid}.enc"
    assert stored.exists()
    assert SECRET.encode() not in stored.read_bytes()
    assert task_store.get_task(tid)["note"] == SECRET  # type: ignore[index]


def test_plain_results_from_before_the_key_are_migrated_on_next_write(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tid = _task_id()
    task_store.update_task_status(tid, "processing", note=SECRET)
    monkeypatch.setattr(settings, "TASK_ENCRYPTION_KEY", Fernet.generate_key().decode())

    assert task_store.get_task(tid)["note"] == SECRET  # type: ignore[index]
    task_store.update_task_status(tid, "completed")

    assert not (data_dir / "tasks" / f"{tid}.json").exists()
    assert (data_dir / "tasks" / f"{tid}.enc").exists()


def test_a_wrong_key_reads_nothing_rather_than_failing(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "TASK_ENCRYPTION_KEY", Fernet.generate_key().decode())
    tid = _task_id()
    task_store.update_task_status(tid, "completed", note=SECRET)

    monkeypatch.setattr(settings, "TASK_ENCRYPTION_KEY", Fernet.generate_key().decode())
    assert task_store.get_task(tid) is None


def test_a_malformed_key_stops_startup(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "TASK_ENCRYPTION_KEY", "not-a-key")
    with pytest.raises(RuntimeError, match="TASK_ENCRYPTION_KEY"):
        task_store.check_encryption_key()


@pytest.mark.parametrize("bad_id", ["..", "..\\..\\secret", "not-a-uuid", ""])
def test_task_ids_that_are_not_uuids_read_nothing(data_dir: Path, bad_id: str) -> None:
    (data_dir / "secret.json").write_text('{"leaked": true}', encoding="utf-8")
    assert task_store.get_task(bad_id) is None


def test_retention_deletes_only_old_results(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, fresh = _task_id(), _task_id()
    task_store.update_task_status(old, "completed")
    task_store.update_task_status(fresh, "completed")
    upload = data_dir / "uploads" / f"{old}.pdf"
    upload.write_bytes(b"%PDF")
    ten_days_ago = time.time() - 10 * 86400
    os.utime(data_dir / "tasks" / f"{old}.json", (ten_days_ago, ten_days_ago))

    assert task_store.purge_expired_results() == 0  # retention off by default

    monkeypatch.setattr(settings, "RESULT_RETENTION_DAYS", 7)
    assert task_store.purge_expired_results() == 1
    assert task_store.get_task(old) is None
    assert task_store.get_task(fresh) is not None
    assert upload.exists()  # uploads are not retention's business


def test_crash_recovery_also_covers_encrypted_tasks(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "TASK_ENCRYPTION_KEY", Fernet.generate_key().decode())
    tid = _task_id()
    task_store.update_task_status(tid, "processing")

    assert task_store.fail_interrupted_tasks() == 1
    assert task_store.get_task(tid)["status"] == "failed"  # type: ignore[index]
