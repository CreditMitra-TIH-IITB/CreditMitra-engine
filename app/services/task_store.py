"""Task persistence: one file per task under DATA_DIR/tasks.

Results hold a borrower's whole statement — every transaction, payee and the
credit report — so two protections are available, both opt-in so an existing
install is never changed underneath its owner:

* Encryption at rest (`TASK_ENCRYPTION_KEY`, a Fernet key). Encrypted results
  are written as `<task_id>.enc`; plain `<task_id>.json` files from before the
  key was set stay readable and are re-written encrypted the next time they
  change.
* Retention (`RESULT_RETENTION_DAYS`). Results older than that are deleted at
  startup and before each new upload. Uploaded PDFs are not touched here — they
  are deleted when their task finishes.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings

logger = logging.getLogger(__name__)

_PLAIN = ".json"
_ENCRYPTED = ".enc"


def _tasks_dir() -> Path:
    return Path(settings.DATA_DIR) / "tasks"


def _fernet() -> Fernet | None:
    key = settings.TASK_ENCRYPTION_KEY.strip()
    return Fernet(key.encode()) if key else None


def check_encryption_key() -> None:
    """Fail fast at startup on a malformed key, rather than at the first save
    — where a task would fail, or worse, results would silently go unencrypted."""
    try:
        _fernet()
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "TASK_ENCRYPTION_KEY is not a valid Fernet key. Generate one with: python -c "
            '"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        ) from exc


def _valid_task_id(task_id: str) -> bool:
    # Task ids are UUIDs minted by the upload endpoint. Anything else reaching
    # here came from a URL and must not become part of a file path.
    try:
        return str(uuid.UUID(task_id)) == task_id
    except (ValueError, AttributeError, TypeError):
        return False


def get_task_file_path(task_id: str) -> Path:
    """Where the task is (or would be) stored under the current settings."""
    suffix = _ENCRYPTED if _fernet() else _PLAIN
    return _tasks_dir() / f"{task_id}{suffix}"


def _read(path: Path) -> dict[str, Any] | None:
    raw = path.read_bytes()
    if path.suffix == _ENCRYPTED:
        fernet = _fernet()
        if fernet is None:
            logger.error(f"{path.name} is encrypted but TASK_ENCRYPTION_KEY is not set")
            return None
        try:
            raw = fernet.decrypt(raw)
        except InvalidToken:
            logger.error(f"{path.name} cannot be decrypted with the configured key")
            return None
    result: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return result


def save_task(task_id: str, data: dict[str, Any]) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    fernet = _fernet()
    path = get_task_file_path(task_id)
    path.write_bytes(fernet.encrypt(payload) if fernet else payload)
    if fernet:
        # A plain copy from before encryption was enabled must not outlive it.
        (_tasks_dir() / f"{task_id}{_PLAIN}").unlink(missing_ok=True)


def get_task(task_id: str) -> dict[str, Any] | None:
    if not _valid_task_id(task_id):
        return None
    for suffix in (_ENCRYPTED, _PLAIN):
        path = _tasks_dir() / f"{task_id}{suffix}"
        if path.exists():
            return _read(path)
    return None


def update_task_status(task_id: str, status: str, **kwargs: Any) -> None:
    task_data = get_task(task_id)
    if not task_data:
        task_data = {"task_id": task_id}
    task_data["status"] = status
    for k, v in kwargs.items():
        task_data[k] = v
    save_task(task_id, task_data)


def _task_files() -> list[Path]:
    tasks_dir = _tasks_dir()
    return [*tasks_dir.glob(f"*{_PLAIN}"), *tasks_dir.glob(f"*{_ENCRYPTED}")]


INTERRUPTED_ERROR = (
    "The engine stopped before this statement finished processing. Please upload it again."
)


def fail_interrupted_tasks() -> int:
    """Mark tasks left pending/processing by a previous process as failed.

    Tasks run inside the engine process, so at startup any task still in
    flight died with the last one (a crash, a kill, a restart). Left alone it
    stays "processing" forever and the dashboard polls it indefinitely. Its
    upload — a bank statement the crash kept from being cleaned up — is deleted
    too. Uploads belonging to finished tasks are not touched.
    """
    uploads_dir = Path(settings.DATA_DIR) / "uploads"
    recovered = 0
    for path in _task_files():
        try:
            data = _read(path)
        except (OSError, ValueError):
            continue
        if not data or data.get("status") not in ("pending", "processing"):
            continue
        task_id = path.stem
        data["status"] = "failed"
        data["error"] = INTERRUPTED_ERROR
        save_task(task_id, data)
        for upload in uploads_dir.glob(f"{task_id}.*"):
            upload.unlink(missing_ok=True)
        recovered += 1
    return recovered


def purge_expired_results() -> int:
    """Delete task results older than RESULT_RETENTION_DAYS (0 keeps them all).

    Age is the file's last write, i.e. when the task last changed state. Only
    result files are removed; uploads are handled by the task lifecycle.
    """
    days = settings.RESULT_RETENTION_DAYS
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for path in _task_files():
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def get_engine_stats() -> dict[str, Any]:
    """Scan stored tasks to calculate total statements processed, total transactions,
    average turnaround time, and list recent statements."""
    total_statements = 0
    total_txns = 0
    total_time = 0.0
    recent: list[dict[str, Any]] = []

    files = _task_files()
    sorted_files = sorted(
        files, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True
    )

    for path in sorted_files:
        try:
            data = _read(path)
            if not data or data.get("status") != "completed":
                continue
            txns = data.get("transactions") or []
            txn_count = len(txns)
            total_statements += 1
            total_txns += txn_count
            proc_time = data.get("elapsed_seconds") or 0.0
            if proc_time > 0:
                total_time += proc_time

            report = data.get("report") or {}
            if len(recent) < 10:
                recent.append({
                    "task_id": data.get("task_id", path.stem),
                    "status": "completed",
                    "filename": data.get("filename"),
                    "created_at": path.stat().st_mtime,
                    "transaction_count": txn_count,
                    "score": report.get("score"),
                    "band": report.get("band"),
                    "archetype": report.get("archetype"),
                    "processing_time_seconds": round(proc_time, 1) if proc_time else None,
                })
        except Exception:
            continue

    avg_time = round(total_time / total_statements, 1) if total_statements > 0 else 0.0
    return {
        "total_statements_processed": total_statements,
        "total_transactions_analyzed": total_txns,
        "avg_processing_time_seconds": avg_time,
        "recent_statements": recent,
    }
