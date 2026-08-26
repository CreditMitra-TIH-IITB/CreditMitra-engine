"""Adapter for the PKDD'99 Financial Dataset (Berka).

This is, as far as I can establish, the only openly available dataset that has
*transaction-level* rows AND observed loan outcomes for the same accounts —
which is what an incremental-lift experiment needs. Aggregate credit datasets
(UCI Default of Credit Card Clients, Give Me Some Credit, Home Credit) ship
pre-summarised features with no payee or category field, so there is nothing for
merchant enrichment to act on and no ablation to run.

    Source:  https://sorry.vse.cz/~berka/challenge/pkdd1999/
    Files:   trans.asc, loan.asc  (semicolon-separated, quoted)
    Licence: released for the PKDD'99 Discovery Challenge; check terms before
             redistributing the raw files.

Outcome. loan.status is A/B/C/D — A finished-and-repaid, B finished-and-defaulted,
C running-and-current, D running-and-in-debt. Default = {B, D}, the standard
treatment for this dataset.

--- Mapping decisions, stated so they can be challenged -------------------------

k_symbol is the closest thing the dataset has to merchant category. It is a
six-value controlled vocabulary, not a merchant name, so it stands in for the
*output* of enrichment rather than its input. That makes this a weaker test than
Indian UPI data would be: fewer categories, no long tail, no resolution failures.

SANKC. UROK ("sanction interest") is charged when the account runs negative. It
is a bank-generated distress marker, the same class of signal as a bounced ECS,
so its narration is written to include "UNPAID" and it is therefore visible to
Arm A. Withholding it would have starved the cash-flow baseline of the dataset's
clearest distress signal and inflated the apparent merchant lift. This is a
judgment call and a reviewer may reasonably want it re-run the other way —
`treat_sanction_as_distress=False` does that.

VYBER / VYBER KARTOU are cash withdrawals; their narrations say "CASH
WITHDRAWAL" so the engine's existing ATM regex fires. That is a translation, not
a concession to the tooling.

--- Known limitations, for the write-up ----------------------------------------

Czech retail banking, 1993-1998. ~680 loans, of which ~70-80 are defaults. At
that size an AUC confidence interval is wide, and a small delta will not clear
it. Nothing here transfers quantitatively to Indian UPI narrations; the value is
that the *method* runs end to end on real outcomes rather than on personas we
wrote ourselves.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

from app.schemas.statements import Transaction

from .contract import Dataset, LabelledStatement

DEFAULT_STATUSES = {"B", "D"}

# k_symbol -> (category, is_essential, lifestyle_dim, recurring_type)
_K_SYMBOL_MAP: dict[str, tuple[str, bool, str, str]] = {
    "POJISTNE": ("insurance", True, "commitment", "emi_like"),
    "SIPO": ("utilities", True, "essential", "emi_like"),
    "UVER": ("loan_emi", False, "commitment", "emi_like"),
    "SLUZBY": ("other", False, "neutral", "subscription"),
    "UROK": ("other", False, "neutral", "adhoc"),
    "DUCHOD": ("other", False, "neutral", "payout_source"),
    "SANKC. UROK": ("other", False, "neutral", "adhoc"),
}

_K_SYMBOL_ENGLISH = {
    "POJISTNE": "INSURANCE PREMIUM",
    "SIPO": "HOUSEHOLD PAYMENT",
    "UVER": "LOAN PAYMENT",
    "SLUZBY": "STATEMENT CHARGE",
    "UROK": "INTEREST CREDITED",
    "DUCHOD": "OLD AGE PENSION",
    "SANKC. UROK": "SANCTION INTEREST",
}

_OPERATION_ENGLISH = {
    "VYBER KARTOU": "CREDIT CARD CASH WITHDRAWAL",
    "VKLAD": "CASH DEPOSIT",
    "PREVOD Z UCTU": "COLLECTION FROM ANOTHER BANK",
    "VYBER": "CASH WITHDRAWAL",
    "PREVOD NA UCET": "REMITTANCE TO ANOTHER BANK",
}

_CREDIT_TYPES = {"PRIJEM", "CREDIT"}


def _parse_date(raw: str) -> date | None:
    """Berka dates are YYMMDD with a 19xx century."""
    raw = raw.strip()
    if len(raw) != 6 or not raw.isdigit():
        return None
    year = 1900 + int(raw[:2])
    try:
        return date(year, int(raw[2:4]), int(raw[4:6]))
    except ValueError:
        return None


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the PKDD'99 dataset and point --data-dir at "
            "the directory holding trans.asc and loan.asc."
        )
    with path.open(encoding="latin-1", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";", quotechar='"'))


def load(
    data_dir: Path | str, treat_sanction_as_distress: bool = True, min_transactions: int = 20
) -> Dataset:
    data_dir = Path(data_dir)
    transactions_raw = _read(data_dir / "trans.asc")
    loans_raw = _read(data_dir / "loan.asc")

    by_account: dict[str, list[Transaction]] = defaultdict(list)
    for row in transactions_raw:
        txn_date = _parse_date(row.get("date", ""))
        if txn_date is None:
            continue
        try:
            amount = float(row.get("amount", "") or 0)
            balance = float(row.get("balance", "") or 0)
        except ValueError:
            continue

        k_symbol = (row.get("k_symbol") or "").strip().upper()
        operation = (row.get("operation") or "").strip().upper()
        txn_type = (row.get("type") or "").strip().upper()
        direction = "credit" if txn_type in _CREDIT_TYPES else "debit"

        narration_parts = [_OPERATION_ENGLISH.get(operation, operation)]
        if k_symbol:
            narration_parts.append(_K_SYMBOL_ENGLISH.get(k_symbol, k_symbol))
        if k_symbol == "SANKC. UROK" and treat_sanction_as_distress:
            # Bank-generated distress marker; see module docstring.
            narration_parts.append("ACCOUNT UNPAID")
        if row.get("bank"):
            narration_parts.append(f"BANK {row['bank']}")
        narration = "/".join(p for p in narration_parts if p)

        category, is_essential, lifestyle_dim, recurring_type = _K_SYMBOL_MAP.get(
            k_symbol, ("other", False, "neutral", "adhoc")
        )
        # A k_symbol is a counterparty *type*, so treat it as the merchant-ish
        # side; a bare transfer with no symbol is left unresolved, which is what
        # an unenriched row looks like in production too.
        resolved = bool(k_symbol)

        by_account[row["account_id"]].append(
            Transaction(
                date=txn_date.strftime("%d-%m-%Y"),
                particulars=narration,
                deposits=f"{amount:.2f}" if direction == "credit" else "",
                withdrawals=f"{amount:.2f}" if direction == "debit" else "",
                balance=f"{balance:.2f}",
                payee=_K_SYMBOL_ENGLISH.get(k_symbol, "").title() if resolved else "",
                payee_type="merchant" if resolved else "person",
                payee_confidence=0.9 if resolved else None,
                txn_date=txn_date,
                amount=amount,
                direction=direction,
                balance_val=balance,
                category=category if resolved else None,
                is_essential=is_essential if resolved else None,
                risk_flag=None,
                lifestyle_dim=lifestyle_dim if resolved else None,  # type: ignore[arg-type]
                recurring_type=recurring_type if resolved else None,  # type: ignore[arg-type]
            )
        )

    statements: list[LabelledStatement] = []
    skipped = 0
    for loan in loans_raw:
        account_id = loan["account_id"]
        rows = by_account.get(account_id, [])
        if len(rows) < min_transactions:
            skipped += 1
            continue
        status = (loan.get("status") or "").strip().upper()
        if status not in {"A", "B", "C", "D"}:
            skipped += 1
            continue
        rows.sort(key=lambda t: t.txn_date or date.min)
        statements.append(
            LabelledStatement(
                borrower_id=f"loan-{loan['loan_id']}",
                transactions=rows,
                defaulted=status in DEFAULT_STATUSES,
                meta={"status": status, "account_id": account_id},
            )
        )

    caveats = (
        "PKDD'99 Berka: Czech retail banking, 1993-1998. Does not transfer "
        "quantitatively to Indian UPI narrations.",
        "k_symbol is a six-value controlled vocabulary standing in for enrichment "
        "output; there is no merchant long tail and no resolution failure to model.",
        f"{len(statements)} loans used, {skipped} skipped for missing or short histories.",
        f"Sanction interest treated as a cash-flow distress marker: {treat_sanction_as_distress}.",
    )
    return Dataset(name="pkdd99-berka", statements=statements, caveats=caveats)
