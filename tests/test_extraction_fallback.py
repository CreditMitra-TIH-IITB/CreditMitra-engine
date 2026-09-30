"""extract_transactions: Docling -> pdfplumber fallback, and failing loudly.

The table readers are monkeypatched so these run without a PDF or Docling
models; what is under test is the control flow around them.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.services import extraction

HEADER = ["Date", "Narration", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"]
ROW = ["01/11/25", "UPI/DR/512345678901/SWIGGY/YESB/swiggy@ybl/Payment", "237.00", "", "4,123.59"]
SUMMARY = [["Account Holder", "Account No"], ["A Customer", "XXXX1234"]]


def _patch(monkeypatch: pytest.MonkeyPatch, docling: object, pdfplumber: object) -> None:
    def fake(result: object) -> Callable[[str], object]:
        def reader(_path: str) -> object:
            if isinstance(result, Exception):
                raise result
            return result

        return reader

    monkeypatch.setattr(extraction, "_docling_tables", fake(docling))
    monkeypatch.setattr(extraction, "_pdfplumber_tables", fake(pdfplumber))


def test_docling_rows_are_parsed_into_transactions(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [[HEADER, ROW]], RuntimeError("must not be called"))
    [txn] = extraction.extract_transactions("statement.pdf")
    assert txn["direction"] == "debit"
    assert txn["amount"] == 237.0
    assert txn["balance_val"] == 4123.59
    assert txn["txn_date"] == "2025-11-01"


def test_empty_docling_result_falls_back_to_pdfplumber(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [SUMMARY], [[HEADER, ROW]])
    assert len(extraction.extract_transactions("statement.pdf")) == 1


def test_docling_crash_falls_back_to_pdfplumber(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, MemoryError("bad_alloc"), [[HEADER, ROW]])
    assert len(extraction.extract_transactions("statement.pdf")) == 1


def test_unrecognised_layout_raises_with_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Previously this returned [] and the task finished as "completed"."""
    _patch(monkeypatch, [SUMMARY], [SUMMARY])
    with pytest.raises(extraction.UnrecognisedStatementError, match="Account Holder"):
        extraction.extract_transactions("statement.pdf")


def test_both_readers_crashing_surfaces_docling_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, ValueError("docling broke"), OSError("pdfplumber broke"))
    with pytest.raises(ValueError, match="docling broke"):
        extraction.extract_transactions("statement.pdf")
