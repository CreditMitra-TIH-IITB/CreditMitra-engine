"""pdfplumber fallback for statements whose header is boxed but whose body is
not (Canara e-passbook). pdfplumber's table finder returns the header alone for
these; the body is rebuilt from word positions."""

from __future__ import annotations

from typing import Any

from app.services.extraction import _borderless_body, _rows_to_records
from app.services.statement_layout import rows_from_tables

# Column x-ranges of the boxed header: Date | Particulars | Deposits | Withdrawals | Balance
COLUMNS = [(10, 103), (103, 288), (288, 400), (400, 492), (492, 585)]
HEADER = ["Date", "Particulars", "Deposits", "Withdrawals", "Balance"]
HEADER_BOTTOM = 50.0


class _Row:
    def __init__(self, cells: list[tuple[float, float, float, float]]) -> None:
        self.cells = cells


class _Table:
    def __init__(self, header: list[str]) -> None:
        self.rows = [_Row([(x0, 30.0, x1, HEADER_BOTTOM) for x0, x1 in COLUMNS])]
        self.bbox = (10.0, 30.0, 585.0, HEADER_BOTTOM)
        self._header = header

    def extract(self) -> list[list[str]]:
        return [self._header]


class _Page:
    def __init__(self, words: list[dict[str, Any]]) -> None:
        self._words = words

    def extract_words(self) -> list[dict[str, Any]]:
        return self._words


def _word(text: str, column: int, top: float) -> dict[str, Any]:
    x0, x1 = COLUMNS[column]
    return {"text": text, "x0": x0 + 2, "x1": min(x0 + 2 + 6 * len(text), x1 - 1), "top": top}


def _transaction(
    top: float, date: str, narration: list[str], amount: tuple[int, str], balance: str
):
    """One Canara-style block: narration lines, then date/amount/balance on the
    last narration line, then a timestamp line."""
    words = [_word(part, 1, top + 12 * i) for i, part in enumerate(narration)]
    anchor = top + 12 * (len(narration) - 1)
    words += [
        _word(date, 0, anchor),
        _word(amount[1], amount[0], anchor),
        _word(balance, 4, anchor),
    ]
    words.append(_word("12:30:45", 1, anchor + 12))
    return words


def _records(page: _Page, table: _Table) -> list[dict[str, Any]]:
    return _rows_to_records(rows_from_tables([_borderless_body(page, table)]))


def test_multiline_transactions_are_rebuilt_from_word_positions() -> None:
    words = [_word("Opening", 1, 60), _word("Balance", 1, 60), _word("10,000.00", 4, 60)]
    # The first transaction sits close under the opening balance line.
    words += _transaction(
        72, "05-11-2025", ["UPI/DR/111/SWIGGY LI/", "UTIB/**GPAY@OK"], (3, "250.00"), "9,750.00"
    )
    words += [_word("Chq:", 1, 130), _word("111", 1, 130)]  # orphan reference line
    words += _transaction(
        154, "06-11-2025", ["UPI/CR/222/RAMESH K/", "SBIN/**RK@OKSBI"], (2, "1,000.00"), "10,750.00"
    )

    records = _records(_Page(words), _Table(HEADER))

    assert [(r["txn_date"], r["direction"], r["amount"], r["balance_val"]) for r in records] == [
        ("2025-11-05", "debit", 250.0, 9750.0),
        ("2025-11-06", "credit", 1000.0, 10750.0),
    ]
    assert records[0]["particulars"].startswith("UPI/DR/111/SWIGGY LI/")


def test_column_roles_come_from_the_header_not_their_position() -> None:
    order = [0, 1, 4, 2, 3]  # Balance printed before Deposits/Withdrawals
    header = [HEADER[i] for i in order]
    words = _transaction(
        60, "05-11-2025", ["UPI/DR/111/SWIGGY LI/"], (order.index(3), "250.00"), "9,750.00"
    )
    # Balance sits under the third column in this layout.
    for w in words:
        if w["text"] == "9,750.00":
            w.update(_word("9,750.00", 2, w["top"]))

    records = _records(_Page(words), _Table(header))

    assert len(records) == 1
    assert records[0]["direction"] == "debit"
    assert records[0]["amount"] == 250.0
    assert records[0]["balance_val"] == 9750.0


def test_a_page_with_no_transactions_yields_nothing() -> None:
    footer = [_word("This", 1, 700), _word("is", 1, 700), _word("computer", 1, 700)]
    assert _borderless_body(_Page(footer), _Table(HEADER)) == []
