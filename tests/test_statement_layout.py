"""Column detection across statement layouts (app/services/statement_layout.py).

Every table here is synthetic. The header strings are the kind of variation
Indian bank exports actually use; before this module, any header not spelled
exactly "withdrawals"/"debit"/"withdrawal" (etc.) produced blank amounts and
the whole statement silently extracted to zero transactions.
"""

from __future__ import annotations

import pytest

from app.services.parsing import derive_direction
from app.services.statement_layout import (
    AMOUNT,
    BALANCE,
    DATE,
    DEPOSITS,
    PARTICULARS,
    TYPE,
    WITHDRAWALS,
    canonical_column,
    map_columns,
    rows_from_tables,
)

UPI_SWIGGY = "UPI/DR/512345678901/SWIGGY/YESB/swiggy@ybl/Payment"
UPI_SALARY = "NEFT/CR/ACME TECHNOLOGIES PVT LTD/SALARY NOV"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Date", DATE),
        ("Txn Date", DATE),
        ("Transaction Date", DATE),
        ("Particulars", PARTICULARS),
        ("Narration", PARTICULARS),
        ("Description", PARTICULARS),
        ("Transaction Remarks", PARTICULARS),
        ("Details", PARTICULARS),
        ("Withdrawals", WITHDRAWALS),
        ("Withdrawal Amt.", WITHDRAWALS),
        ("Withdrawal (Dr)", WITHDRAWALS),
        ("Debit", WITHDRAWALS),
        ("Debit Amount (INR)", WITHDRAWALS),
        ("Deposits", DEPOSITS),
        ("Deposit Amt.", DEPOSITS),
        ("Deposit (Cr)", DEPOSITS),
        ("Credit", DEPOSITS),
        ("Balance", BALANCE),
        ("Closing Balance", BALANCE),
        ("Balance (INR)", BALANCE),
        ("Amount", AMOUNT),
        ("Amount (INR)", AMOUNT),
        ("Dr/Cr", TYPE),
        ("Cr / Dr", TYPE),
        ("Debit/Credit", TYPE),
        ("Type", TYPE),
        ("Chq./Ref.No.", None),
        ("Ref No./Cheque No.", None),
        ("Sl No", None),
        ("Branch Code", None),
        ("", None),
    ],
)
def test_canonical_column(header: str, expected: str | None) -> None:
    assert canonical_column(header) == expected


def test_transaction_date_beats_value_date_in_either_order() -> None:
    assert map_columns(["Value Date", "Txn Date", "Narration", "Debit"])[DATE] == 1
    assert map_columns(["Txn Date", "Value Date", "Narration", "Debit"])[DATE] == 0


def test_existing_canara_layout_is_unchanged() -> None:
    table = [
        ["Date", "Particulars", "Deposits", "Withdrawals", "Balance"],
        ["05-11-2025", UPI_SWIGGY, "", "237.00", "43,123.59"],
        ["06-11-2025", UPI_SALARY, "45,000.00", "", "88,123.59"],
    ]
    rows = rows_from_tables([table]).rows
    assert len(rows) == 2
    assert rows[0] == {
        "date": "05-11-2025",
        "particulars": UPI_SWIGGY,
        "deposits": "",
        "withdrawals": "237.00",
        "balance": "43,123.59",
    }
    assert derive_direction(rows[1]["deposits"], rows[1]["withdrawals"]) == ("credit", 45000.0)


def test_amt_style_headers_now_extract_amounts() -> None:
    """The layout that used to extract to zero transactions."""
    table = [
        [
            "Date",
            "Narration",
            "Chq./Ref.No.",
            "Value Dt",
            "Withdrawal Amt.",
            "Deposit Amt.",
            "Closing Balance",
        ],
        ["01/11/25", UPI_SWIGGY, "0000512345678901", "01/11/25", "237.00", "", "43,123.59"],
        ["02/11/25", UPI_SALARY, "0000N123456789", "02/11/25", "", "45,000.00", "88,123.59"],
    ]
    rows = rows_from_tables([table]).rows
    assert [derive_direction(r["deposits"], r["withdrawals"]) for r in rows] == [
        ("debit", 237.0),
        ("credit", 45000.0),
    ]
    assert rows[0]["balance"] == "43,123.59"


def test_single_amount_column_with_type_column() -> None:
    table = [
        ["Txn Date", "Description", "Amount (INR)", "Dr/Cr", "Balance (INR)"],
        ["01-11-2025", UPI_SWIGGY, "237.00", "DR", "43,123.59"],
        ["02-11-2025", UPI_SALARY, "45,000.00", "CR", "88,123.59"],
    ]
    rows = rows_from_tables([table]).rows
    assert [derive_direction(r["deposits"], r["withdrawals"]) for r in rows] == [
        ("debit", 237.0),
        ("credit", 45000.0),
    ]


def test_single_amount_column_with_suffix() -> None:
    table = [
        ["Date", "Particulars", "Amount", "Balance"],
        ["01-11-2025", UPI_SWIGGY, "237.00 Dr", "43,123.59 Cr"],
        ["02-11-2025", UPI_SALARY, "45,000.00 Cr", "88,123.59 Cr"],
    ]
    rows = rows_from_tables([table]).rows
    assert [derive_direction(r["deposits"], r["withdrawals"]) for r in rows] == [
        ("debit", 237.0),
        ("credit", 45000.0),
    ]


def test_single_amount_with_unknown_direction_is_left_blank() -> None:
    """No type column and no suffix: guessing a sign would silently flip
    income into spending, so the row carries no amount instead."""
    table = [["Date", "Particulars", "Amount"], ["01-11-2025", UPI_SWIGGY, "237.00"]]
    row = rows_from_tables([table]).rows[0]
    assert derive_direction(row["deposits"], row["withdrawals"]) is None


def test_continuation_page_without_header_reuses_layout() -> None:
    page1 = [
        ["Date", "Narration", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"],
        ["01/11/25", UPI_SWIGGY, "237.00", "", "43,123.59"],
    ]
    page2 = [["02/11/25", UPI_SALARY, "", "45,000.00", "88,123.59"]]
    rows = rows_from_tables([page1, page2]).rows
    assert len(rows) == 2
    assert rows[1]["deposits"] == "45,000.00"


def test_repeated_header_on_continuation_page_is_skipped() -> None:
    header = ["Date", "Narration", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"]
    page1 = [header, ["01/11/25", UPI_SWIGGY, "237.00", "", "43,123.59"]]
    page2 = [
        ["DATE", "NARRATION", "WITHDRAWAL AMT.", "DEPOSIT AMT.", "CLOSING BALANCE"],
        ["02/11/25", UPI_SALARY, "", "45,000.00", "88,123.59"],
    ]
    rows = rows_from_tables([page1, page2]).rows
    assert [r["date"] for r in rows] == ["01/11/25", "02/11/25"]


def test_wrapped_narration_joins_previous_row() -> None:
    table = [
        ["Date", "Particulars", "Deposits", "Withdrawals", "Balance"],
        ["05-11-2025", "UPI/DR/512345678901/SWIG", "", "237.00", "43,123.59"],
        ["", "GY/YESB/swiggy@ybl/Payment", "", "", ""],
        ["06-11-2025", UPI_SALARY, "45,000.00", "", "88,123.59"],
    ]
    rows = rows_from_tables([table]).rows
    assert len(rows) == 2
    assert rows[0]["particulars"] == "UPI/DR/512345678901/SWIGGY/YESB/swiggy@ybl/Payment"


def test_orphan_chq_row_is_not_merged_into_previous_narration() -> None:
    """Canara emits standalone 'Chq: <ref>' rows; they are junk, not a
    continuation, and must not be glued onto the previous narration."""
    table = [
        ["Date", "Particulars", "Deposits", "Withdrawals", "Balance"],
        ["05-11-2025", UPI_SWIGGY, "", "237.00", "43,123.59"],
        ["", "Chq: 101895374870", "", "", ""],
    ]
    rows = rows_from_tables([table]).rows
    assert rows[0]["particulars"] == UPI_SWIGGY


def test_unreadable_table_reports_its_header() -> None:
    table = [["Account No", "Branch", "IFSC"], ["XXXX1234", "Powai", "XXXX0001234"]]
    result = rows_from_tables([table])
    assert result.rows == []
    assert result.unrecognised_headers == [["Account No", "Branch", "IFSC"]]


def test_summary_table_before_transactions_does_not_block_layout() -> None:
    """Statements often open with an account-summary table; it must be
    reported and skipped, not adopted as the layout for what follows."""
    summary = [["Account Holder", "Account No"], ["A Customer", "XXXX1234"]]
    txns = [
        ["Date", "Narration", "Debit", "Credit", "Balance"],
        ["01/11/25", UPI_SWIGGY, "237.00", "", "43,123.59"],
    ]
    result = rows_from_tables([summary, txns])
    assert len(result.rows) == 1
    assert result.unrecognised_headers == [["Account Holder", "Account No"]]


def test_trailing_totals_table_is_not_read_as_transactions() -> None:
    txns = [
        ["Date", "Narration", "Debit", "Credit", "Balance"],
        ["01/11/25", UPI_SWIGGY, "237.00", "", "43,123.59"],
    ]
    totals = [
        ["Opening Balance", "Total Debits", "Total Credits", "Closing Balance"],
        ["43,360.59", "237.00", "0.00", "43,123.59"],
    ]
    result = rows_from_tables([txns, totals])
    assert len(result.rows) == 1
