"""The uploaded statement must be deleted once its task finishes, either way."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.services import extraction, merchant_classifier


@pytest.fixture()
def statuses(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    recorded: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        extraction,
        "update_task_status",
        lambda task_id, status, **kw: recorded.append((status, kw)),
    )
    return recorded


def _upload(tmp_path: Path) -> Path:
    pdf = tmp_path / "statement.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf


def test_upload_deleted_when_task_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statuses: list[Any]
) -> None:
    def boom(_: str) -> list[dict[str, Any]]:
        raise ValueError("unreadable statement")

    monkeypatch.setattr(extraction, "extract_transactions", boom)
    pdf = _upload(tmp_path)

    extraction.process_pdf_task("t-fail", str(pdf))

    assert not pdf.exists()
    assert statuses[-1][0] == "failed"


def test_upload_deleted_when_task_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statuses: list[Any]
) -> None:
    class _NoModelClassifier:
        def classify_batch(self, names: list[str]) -> list[dict[str, Any]]:
            return [{"label": None, "confidence": None} for _ in names]

    monkeypatch.setattr(extraction, "extract_transactions", lambda _: [])
    monkeypatch.setattr(
        merchant_classifier.MerchantClassifierService,
        "get_instance",
        classmethod(lambda cls: _NoModelClassifier()),
    )
    pdf = _upload(tmp_path)

    extraction.process_pdf_task("t-ok", str(pdf))

    assert not pdf.exists()
    assert statuses[-1][0] == "completed"
