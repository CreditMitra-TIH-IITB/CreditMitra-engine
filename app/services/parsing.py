"""Amount / date / direction parsing for Indian bank statements.

ISSUE #5. Pure functions, no I/O. Docling gives strings; the risk brain needs
numbers.

Written against REAL formats observed in Canara Bank and HDFC statements:
    amounts : "43,123.59"  "1,524.05"  "1,23,456.78" (lakh grouping)
              "193.00 Cr"  "(1,000.00)"  "₹237.00"  "-"  ""
    dates   : "05-11-2025"  "01/11/25"  "06 Nov 2025"  "06-Nov-25"
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

Direction = Literal["credit", "debit"]

# ---------------------------------------------------------------------------
# Amounts
# ---------------------------------------------------------------------------

# Strip everything that isn't a digit, dot, or minus. Handles Indian lakh
# grouping (1,23,456.78) for free because we just drop all commas.
_AMOUNT_STRIP_RE = re.compile(r"[^\d.\-]")
_CR_SUFFIX_RE = re.compile(r"\bCR\b|\bCr\b", re.IGNORECASE)
_DR_SUFFIX_RE = re.compile(r"\bDR\b|\bDr\b", re.IGNORECASE)
_BLANKS = {"", "-", "--", "n/a", "na", "nil", "none"}


def parse_amount(s: str | None) -> float | None:
    """'1,23,456.78' -> 123456.78 ; '(1,000.00)' -> -1000.0 ; '-' -> None.

    Returns None (never raises) for blanks, junk, or unparseable input —
    callers treat None as "no amount on this row".
    """
    if s is None:
        return None
    raw = str(s).strip()
    if raw.lower() in _BLANKS:
        return None

    # Parenthesised negatives: (1,000.00)
    negative = raw.startswith("(") and raw.endswith(")")

    # Note: we deliberately drop Cr/Dr here. Sign comes from which COLUMN the
    # value sits in (see derive_direction), not from the suffix — mixing the
    # two is how you get double-negatives.
    cleaned = _AMOUNT_STRIP_RE.sub("", raw)
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None

    # Guard against multiple dots from OCR noise ("1.234.56")
    if cleaned.count(".") > 1:
        head, _, tail = cleaned.rpartition(".")
        cleaned = head.replace(".", "") + "." + tail

    try:
        value = float(cleaned)
    except ValueError:
        return None

    if negative:
        value = -abs(value)
    return value


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

# Order matters: most specific / most common first. dd before mm throughout —
# Indian statements are day-first, never US month-first.
_DATE_FORMATS = (
    "%d-%m-%Y",  # 05-11-2025   (Canara)
    "%d/%m/%Y",  # 05/11/2025
    "%d-%m-%y",  # 05-11-25
    "%d/%m/%y",  # 01/11/25     (HDFC)
    "%d-%b-%Y",  # 06-Nov-2025
    "%d-%b-%y",  # 06-Nov-25
    "%d %b %Y",  # 06 Nov 2025
    "%d %b %y",  # 06 Nov 25
    "%d-%B-%Y",  # 06-November-2025
    "%Y-%m-%d",  # 2025-11-06   (ISO, just in case)
)

_DATE_CLEAN_RE = re.compile(r"[^\dA-Za-z\-/ ]")


def parse_date(s: str | None) -> date | None:
    """'05-11-2025' -> date(2025,11,5). Day-first. None on failure."""
    if s is None:
        return None
    raw = _DATE_CLEAN_RE.sub("", str(s).strip())
    if not raw or raw.lower() in _BLANKS:
        return None

    # Some rows carry a trailing time: "05-11-2025 22:28:03"
    raw = raw.split(" ", 1)[0] if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s", raw) else raw

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------


def derive_direction(
    deposits: str | None,
    withdrawals: str | None,
) -> tuple[Direction, float] | None:
    """Which column is populated decides credit vs debit.

    Canara puts the value in `deposits` OR `withdrawals`, never both.
    Returns None when neither column has a number (header rows, Chq: rows,
    Opening/Closing Balance rows).
    """
    dep = parse_amount(deposits)
    wdr = parse_amount(withdrawals)

    # Both populated shouldn't happen; if it does, the larger wins.
    if dep is not None and wdr is not None:
        if abs(dep) >= abs(wdr):
            return ("credit", abs(dep))
        return ("debit", abs(wdr))

    if dep is not None:
        return ("credit", abs(dep))
    if wdr is not None:
        return ("debit", abs(wdr))
    return None


def derive_direction_from_type(type_str: str | None) -> Direction | None:
    """Fallback for statements with a DR/CR type column instead of two
    amount columns (the HDFC e-passbook export uses this)."""
    if not type_str:
        return None
    t = str(type_str).strip().upper()
    if t in {"CR", "CREDIT", "C"}:
        return "credit"
    if t in {"DR", "DEBIT", "D"}:
        return "debit"
    return None


# ---------------------------------------------------------------------------
# Row filtering  (extraction hardening — see below)
# ---------------------------------------------------------------------------

_SUMMARY_ROW_RE = re.compile(r"^\s*(opening|closing)\s+balance\s*$", re.IGNORECASE)
# Canara emits standalone "Chq: 101895374870" rows and also prepends the
# previous row's ref into the next row's particulars. Both produce garbage
# payees ("ABC Bank Ltd" hallucinations) if sent to the SLM.
_CHQ_ONLY_RE = re.compile(r"^\s*Chq:\s*\d+\s*$", re.IGNORECASE)
_CHQ_PREFIX_RE = re.compile(r"^\s*Chq:\s*\d+\s+", re.IGNORECASE)
_CHQ_SUFFIX_RE = re.compile(r"\s+Chq:\s*\d+\s*$", re.IGNORECASE)


def is_summary_row(row: dict) -> bool:
    """Opening/Closing Balance rows — note Canara puts the LABEL in the
    deposits column, not particulars, so checking particulars alone misses it.
    """
    for field in ("particulars", "deposits", "withdrawals", "date"):
        val = str(row.get(field, "") or "")
        if _SUMMARY_ROW_RE.match(val):
            return True
    return False


def is_junk_row(row: dict) -> bool:
    """Rows with no real narration — orphan 'Chq: <ref>' lines, empty rows.
    These must never reach the payee model; it hallucinates bank names.
    """
    if is_summary_row(row):
        return True
    particulars = str(row.get("particulars", "") or "").strip()
    if not particulars:
        return True
    return bool(_CHQ_ONLY_RE.match(particulars))


def strip_chq_artifacts(narration: str) -> str:
    """Remove the leading/trailing 'Chq: <digits>' bleed from a narration.

    Canara's table extraction leaks the previous row's reference number into
    the next row:
        "Chq: 740828843104 UPI/DR/101899321560/IRCTCT OUR/..."
    -> "UPI/DR/101899321560/IRCTCT OUR/..."
    """
    s = narration or ""
    s = _CHQ_PREFIX_RE.sub("", s)
    s = _CHQ_SUFFIX_RE.sub("", s)
    return s.strip()
