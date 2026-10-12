"""Render synthetic bank statements in different banks' table layouts.

    python scripts/make_sample_statements.py [out_dir]

The transactions are the synthetic persona fixtures (tests/fixtures/personas) —
no real customer data. What varies is the *layout*: header names, column order,
date format, and whether amounts sit in Dr/Cr columns or one signed column.
That is what breaks extraction across banks, so these give an end-to-end check
of every layout we claim to read without anyone's real statement.

Requires reportlab (dev-only; not an engine dependency).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = ROOT / "tests" / "fixtures" / "personas"

Row = dict[str, Any]


def _d(row: Row) -> date:
    return datetime.strptime(row["date"], "%d-%m-%Y").date()


def _ref(i: int) -> str:
    return f"{400000000000 + i * 7919:012d}"


# Each layout: (bank title, header, row renderer)
def _hdfc(i: int, r: Row) -> list[str]:
    return [
        _d(r).strftime("%d/%m/%y"),
        r["particulars"],
        _ref(i),
        _d(r).strftime("%d/%m/%y"),
        r["withdrawals"],
        r["deposits"],
        r["balance"],
    ]


def _sbi(i: int, r: Row) -> list[str]:
    return [
        _d(r).strftime("%d %b %Y"),
        _d(r).strftime("%d %b %Y"),
        r["particulars"],
        _ref(i),
        r["withdrawals"],
        r["deposits"],
        r["balance"],
    ]


def _icici(i: int, r: Row) -> list[str]:
    return [
        str(i + 1),
        _d(r).strftime("%d/%m/%Y"),
        _d(r).strftime("%d/%m/%Y"),
        "",
        r["particulars"],
        r["withdrawals"] or "0.00",
        r["deposits"] or "0.00",
        r["balance"],
    ]


def _axis(i: int, r: Row) -> list[str]:
    # Single amount column + Dr/Cr type column.
    is_credit = bool(r["deposits"])
    return [
        _d(r).strftime("%d-%m-%Y"),
        _ref(i),
        r["particulars"],
        r["deposits"] if is_credit else r["withdrawals"],
        "CR" if is_credit else "DR",
        r["balance"],
    ]


LAYOUTS: dict[str, tuple[str, list[str], Callable[[int, Row], list[str]], str]] = {
    "hdfc": (
        "HDFC BANK — Statement of Account",
        [
            "Date",
            "Narration",
            "Chq./Ref.No.",
            "Value Dt",
            "Withdrawal Amt.",
            "Deposit Amt.",
            "Closing Balance",
        ],
        _hdfc,
        "02_gig_hustler",
    ),
    "sbi": (
        "STATE BANK OF INDIA — Account Statement",
        [
            "Txn Date",
            "Value Date",
            "Description",
            "Ref No./Cheque No.",
            "Debit",
            "Credit",
            "Balance",
        ],
        _sbi,
        "01_salaried_saver",
    ),
    "icici": (
        "ICICI Bank — Detailed Statement",
        [
            "S No.",
            "Value Date",
            "Transaction Date",
            "Cheque Number",
            "Transaction Remarks",
            "Withdrawal Amount (INR )",
            "Deposit Amount (INR )",
            "Balance (INR )",
        ],
        _icici,
        "03_bnpl_heavy_spender",
    ),
    "axis": (
        "AXIS BANK — Statement of Account",
        ["Tran Date", "Chq No", "Particulars", "Amount", "Dr/Cr", "Balance"],
        _axis,
        "04_gambler",
    ),
}


def render(bank: str, out_dir: Path) -> Path:
    title, header, row_fn, persona = LAYOUTS[bank]
    rows = json.loads((PERSONAS / f"{persona}.json").read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    cell = styles["BodyText"].clone("cell", fontSize=7, leading=8.5)

    data: list[list[Any]] = [header]
    for i, r in enumerate(rows):
        data.append([Paragraph(str(v), cell) for v in row_fn(i, r)])

    out = out_dir / f"synthetic_{bank}_{persona}.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=landscape(A4), leftMargin=10 * mm, rightMargin=10 * mm
    )
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
                ("FONTSIZE", (0, 0), (-1, 0), 7.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    doc.build(
        [
            Paragraph(title, styles["Heading2"]),
            Paragraph("SYNTHETIC TEST STATEMENT — not a real account", styles["Normal"]),
            Spacer(1, 4 * mm),
            table,
        ]
    )
    return out


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "sample_statements"
    out_dir.mkdir(parents=True, exist_ok=True)
    for bank in LAYOUTS:
        print(render(bank, out_dir))


if __name__ == "__main__":
    main()
