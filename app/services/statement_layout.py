"""Column detection for bank-statement tables.

Pure functions, no I/O. Docling and pdfplumber both hand us tables as rows of
strings; this module decides which column is which, so the extractors never
hardcode a bank's header names.

Why this exists: headers used to be matched by exact name ("withdrawals",
"debit", "withdrawal"), so a statement headed "Withdrawal Amt." or
"Withdrawal (Dr)" produced rows with every amount blank. Those rows were then
filtered as junk and the task finished as "completed" with zero transactions —
a silent failure on every layout nobody had hand-listed.

Handled here:
    * keyword matching on normalised headers ("Withdrawal Amt." -> withdrawals,
      "Transaction Remarks" -> particulars, "Balance (INR)" -> balance)
    * transaction date preferred over value date when both exist
    * single "Amount" column + a Dr/Cr type column, or a Cr/Dr suffix on the
      amount itself
    * continuation tables on later pages that repeat the header, or have no
      header row at all (the previous table's layout is carried over)
    * narrations wrapped onto a second row with no date and no amounts
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.services.parsing import (
    derive_direction_from_type,
    is_junk_row,
    parse_amount,
    parse_date,
)

# Canonical fields the rest of the pipeline reads. "amount" and "type" only
# exist on single-amount layouts and are folded into deposits/withdrawals.
DATE = "date"
PARTICULARS = "particulars"
DEPOSITS = "deposits"
WITHDRAWALS = "withdrawals"
BALANCE = "balance"
AMOUNT = "amount"
TYPE = "type"

OUTPUT_FIELDS = (DATE, PARTICULARS, DEPOSITS, WITHDRAWALS, BALANCE)

_NON_WORD_RE = re.compile(r"[^a-z0-9/ ]+")
_SPACE_RE = re.compile(r"\s+")

_PARTICULARS_WORDS = ("narration", "particular", "description", "remark", "details")
_WITHDRAWAL_WORDS = ("withdraw", "debit")
_DEPOSIT_WORDS = ("deposit", "credit")
# Columns that look like they match a keyword but must never be read as one.
_IGNORED_TOKENS = {"chq", "cheque", "ref", "utr", "serial", "branch", "code"}
_IGNORED_HEADERS = {"no", "sl no", "s no", "sr no"}


def normalize_header(header: object) -> str:
    """'Withdrawal Amt. (INR)' -> 'withdrawal amt inr'."""
    text = str(header or "").lower().replace("\n", " ")
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _tokens(norm: str) -> set[str]:
    return set(norm.replace("/", " ").split())


def canonical_column(header: object) -> str | None:
    """Map one raw header to a canonical field, or None if it is not one we read.

    Order matters: a Dr/Cr type column mentions both "debit" and "credit", and
    "Withdrawal (Dr)" mentions "dr", so the type check runs first and the
    two-sided test decides between them.
    """
    norm = normalize_header(header)
    if not norm:
        return None
    tokens = _tokens(norm)

    if "balance" in norm:
        return BALANCE
    if norm in _IGNORED_HEADERS or (tokens & _IGNORED_TOKENS and "date" not in norm):
        return None

    has_debit = bool(tokens & {"dr"}) or any(w in norm for w in _WITHDRAWAL_WORDS)
    has_credit = bool(tokens & {"cr"}) or any(w in norm for w in _DEPOSIT_WORDS)
    if (has_debit and has_credit) or norm in {"type", "txn type", "transaction type"}:
        return TYPE

    if "date" in norm:
        return DATE
    if any(w in norm for w in _PARTICULARS_WORDS):
        return PARTICULARS
    if has_debit:
        return WITHDRAWALS
    if has_credit:
        return DEPOSITS
    if "amount" in norm or "amt" in tokens:
        return AMOUNT
    return None


def map_columns(headers: Sequence[object]) -> dict[str, int]:
    """Canonical field -> column index. First match wins, except that a
    transaction/posting date beats a value date."""
    mapping: dict[str, int] = {}
    for idx, header in enumerate(headers):
        field_name = canonical_column(header)
        if field_name is None:
            continue
        if field_name == DATE and DATE in mapping:
            current = normalize_header(headers[mapping[DATE]])
            if "value" in current and "value" not in normalize_header(header):
                mapping[DATE] = idx
            continue
        mapping.setdefault(field_name, idx)
    return mapping


def is_usable_layout(mapping: dict[str, int]) -> bool:
    """A table we can read needs a narration and some way to get an amount."""
    has_amounts = DEPOSITS in mapping or WITHDRAWALS in mapping or AMOUNT in mapping
    return PARTICULARS in mapping and has_amounts


def _cell(cells: Sequence[object], idx: int | None) -> str:
    if idx is None or idx >= len(cells):
        return ""
    value = cells[idx]
    return "" if value is None else str(value).strip()


def _single_amount_direction(amount: str, type_value: str) -> tuple[str, str]:
    """(deposits, withdrawals) for a single-amount row. Direction comes from
    the type column, else from a Cr/Dr suffix on the amount. Unknown direction
    leaves both blank rather than guessing a sign."""
    direction = derive_direction_from_type(type_value)
    if direction is None:
        upper = amount.upper()
        if re.search(r"\bCR\b", upper):
            direction = "credit"
        elif re.search(r"\bDR\b", upper):
            direction = "debit"
    if direction == "credit":
        return amount, ""
    if direction == "debit":
        return "", amount
    return "", ""


def row_to_fields(cells: Sequence[object], mapping: dict[str, int]) -> dict[str, str]:
    """One table row -> {date, particulars, deposits, withdrawals, balance}."""
    out = {
        DATE: _cell(cells, mapping.get(DATE)),
        PARTICULARS: _cell(cells, mapping.get(PARTICULARS)),
        DEPOSITS: _cell(cells, mapping.get(DEPOSITS)),
        WITHDRAWALS: _cell(cells, mapping.get(WITHDRAWALS)),
        BALANCE: _cell(cells, mapping.get(BALANCE)),
    }
    if AMOUNT in mapping and not out[DEPOSITS] and not out[WITHDRAWALS]:
        out[DEPOSITS], out[WITHDRAWALS] = _single_amount_direction(
            _cell(cells, mapping.get(AMOUNT)), _cell(cells, mapping.get(TYPE))
        )
    return out


def _is_wrapped_narration(fields: dict[str, str]) -> bool:
    """A second line of the previous row's narration: text but no date and no
    numbers. Sending it to the payee model on its own produces a garbage payee.
    Orphan "Chq: <ref>" and Opening/Closing Balance rows are junk, not
    continuations — merging them would glue a reference number onto the
    previous narration."""
    if not fields[PARTICULARS] or is_junk_row(fields):
        return False
    if parse_date(fields[DATE]) is not None:
        return False
    return all(parse_amount(fields[f]) is None for f in (DEPOSITS, WITHDRAWALS, BALANCE))


@dataclass
class TableParseResult:
    rows: list[dict[str, str]] = field(default_factory=list)
    # Header rows of tables that could not be mapped — surfaced in the error
    # message when nothing at all was extracted, so an unsupported layout is
    # diagnosable from the task record alone.
    unrecognised_headers: list[list[str]] = field(default_factory=list)


def rows_from_tables(tables: Iterable[Sequence[Sequence[object]]]) -> TableParseResult:
    """Walk every table of a statement in page order and return field dicts.

    Each table is a list of rows; its first row is treated as a header when it
    maps to a usable layout. Otherwise the previous table's layout is reused,
    because statements that span pages usually print the header only once.
    """
    result = TableParseResult()
    layout: dict[str, int] | None = None
    header_norms: list[str] = []

    for table in tables:
        table_rows = [list(r) for r in table if r is not None]
        if not table_rows:
            continue

        candidate = map_columns(table_rows[0])
        if is_usable_layout(candidate):
            layout = candidate
            header_norms = [normalize_header(h) for h in table_rows[0]]
            body = table_rows[1:]
        elif layout is not None and len(table_rows[0]) == len(header_norms):
            # Same width as the last header: a continuation page. A table of a
            # different width (account summary, totals) is not, and reading it
            # with the transaction layout would invent transactions.
            body = table_rows
        else:
            result.unrecognised_headers.append([str(h or "").strip() for h in table_rows[0]])
            continue

        for cells in body:
            # Header repeated at the top of a continuation page.
            if [normalize_header(c) for c in cells] == header_norms:
                continue
            fields = row_to_fields(cells, layout)
            if not any(fields.values()):
                continue
            if _is_wrapped_narration(fields) and result.rows:
                prev = result.rows[-1]
                prev[PARTICULARS] = f"{prev[PARTICULARS]}{fields[PARTICULARS]}"
                continue
            result.rows.append(fields)

    return result
