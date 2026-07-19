"""Cash-flow feature extraction for the risk brain (Track B).

ISSUE #10. Pure function over parsed, enriched transactions -> FeatureVector.
No I/O, no LLM calls — everything here is arithmetic over fields Issues #5
(parsing) and #7 (enrichment) already populated.

FAIR LENDING (docs/taxonomy.md): healthcare must never lower a credit score.
Every ratio in this module excludes healthcare transactions from BOTH the
numerator and the denominator — not just zero-weighted, fully absent — so a
month with a large medical bill can't even dilute an unrelated ratio like
essential_ratio or merchant_resolution_rate.

GIG INCOME (docs/taxonomy.md "same merchant, two directions"): the merchant
dictionary stores the DEBIT meaning for gig platforms (Swiggy = food_delivery
when you pay it). A CREDIT from the same merchant is gig income, reinterpreted
here — not at enrichment time — via
MerchantEnrichmentService.is_gig_payout_source().
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from app.schemas.statements import FeatureVector, Transaction
from app.services.merchant_enrichment import get_enrichment_service

# ---------------------------------------------------------------------------
# Narration pattern matching — signals no enrichment tier can see, because
# there's no merchant name to look up (ATM withdrawals) or the signal lives
# in bank-generated boilerplate, not a payee (bounced payments).
# ---------------------------------------------------------------------------

_CASH_WITHDRAWAL_RE = re.compile(
    r"\bATM\b|\bCASH\s*W(?:I)?T?H?D(?:RAWAL)?\b|\bCSH\s*WDL\b", re.IGNORECASE
)
_BOUNCE_RE = re.compile(
    r"\bRETURN(?:ED)?\b|\bBOUNCE[D]?\b|\bECS\s*RET\b|\bCHQ\s*RET\b|"
    r"\bINSUFFICIENT\s*FUND|\bUNPAID\b|\bDISHONO[UR]*R",
    re.IGNORECASE,
)

_EMI_LIKE_RECURRING = {"emi_like"}
_ASPIRATIONAL_DIM = "aspirational"
_ESSENTIAL_DIM = "essential"

# Relative amount tolerance for treating two credits as "the same recurring
# payment" (salary, gig payout) rather than coincidentally similar amounts.
_RECURRENCE_AMOUNT_TOLERANCE = 0.20


def _month_key(d: date) -> tuple[int, int]:
    return (d.year, d.month)


def _usable(transactions: list[Transaction]) -> list[Transaction]:
    """Rows with enough parsed data to reason about at all."""
    return [t for t in transactions if t.txn_date is not None and t.amount is not None]


def _scoring_relevant(transactions: list[Transaction]) -> list[Transaction]:
    """Usable rows, minus healthcare — the fair-lending exclusion applied
    uniformly so every ratio in this module treats it the same way."""
    return [t for t in _usable(transactions) if t.category != "healthcare"]


def _debits(transactions: list[Transaction]) -> list[Transaction]:
    return [t for t in transactions if t.direction == "debit"]


def _credits(transactions: list[Transaction]) -> list[Transaction]:
    return [t for t in transactions if t.direction == "credit"]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _detect_recurring_income(credits: list[Transaction]) -> tuple[bool, float]:
    """Group credits by normalized payee; a payee with >=2 credits across
    >=2 distinct months, amounts within tolerance of each other, is treated
    as a salary-like recurring income source. Returns (found, avg_amount).

    Restricted to non-merchant payees (an employer, not a business). A gig
    platform (Swiggy/Uber/Ola/...) can just as easily pay out on a
    recurring monthly-ish cadence — that's gig income, handled separately
    by _detect_gig_income, and must not get mislabelled as "salary".
    """
    by_payee: dict[str, list[Transaction]] = defaultdict(list)
    for t in credits:
        if t.payee_type == "merchant":
            continue
        key = (t.payee or "").strip().lower()
        if key:
            by_payee[key].append(t)

    best_avg = 0.0
    best_total = 0.0
    for txns in by_payee.values():
        months = {_month_key(t.txn_date) for t in txns if t.txn_date}
        if len(txns) < 2 or len(months) < 2:
            continue
        amounts = [t.amount for t in txns if t.amount is not None]
        if not amounts:
            continue
        avg = sum(amounts) / len(amounts)
        if avg <= 0:
            continue
        if all(abs(a - avg) / avg <= _RECURRENCE_AMOUNT_TOLERANCE for a in amounts):
            total = sum(amounts)
            if total > best_total:
                best_total, best_avg = total, avg

    return (best_avg > 0, best_avg)


def _detect_gig_income(credits: list[Transaction]) -> float:
    """Average monthly credit amount from merchants flagged as gig payout
    sources (Swiggy/Uber/Ola/... paying the user), when it recurs across
    >=2 distinct months. See module docstring."""
    service = get_enrichment_service()
    gig_credits = [
        t for t in credits if t.payee and service.is_gig_payout_source(t.payee) and t.amount
    ]
    months = {_month_key(t.txn_date) for t in gig_credits if t.txn_date}
    if len(gig_credits) < 2 or len(months) < 2:
        return 0.0
    total = sum(t.amount for t in gig_credits if t.amount is not None)
    return total / len(months)


def build_features(transactions: list[Transaction]) -> FeatureVector:
    """Compute cash-flow features from a statement's parsed, enriched rows.

    Never raises — a statement with sparse/missing parsed fields just
    produces a mostly-zero FeatureVector rather than blocking the pipeline.
    """
    usable = _usable(transactions)
    scoring = _scoring_relevant(transactions)

    months = {_month_key(t.txn_date) for t in usable if t.txn_date}
    months_covered = len(months)

    credits = _credits(scoring)
    debits = _debits(scoring)

    salary_detected, salary_avg = _detect_recurring_income(credits)
    gig_avg = _detect_gig_income(_credits(usable))  # gig credits may be non-essential-scored too

    if salary_detected:
        monthly_income = salary_avg
    elif gig_avg > 0:
        monthly_income = gig_avg
    else:
        total_credit = sum(t.amount for t in credits if t.amount is not None)
        monthly_income = _safe_div(total_credit, max(months_covered, 1))

    total_debit = sum(t.amount for t in debits if t.amount is not None)
    total_credit_all = sum(t.amount for t in credits if t.amount is not None)
    net_cashflow = total_credit_all - total_debit

    emi_like_debit = sum(
        t.amount for t in debits if t.amount is not None and t.recurring_type in _EMI_LIKE_RECURRING
    )
    foir = _safe_div(emi_like_debit, monthly_income * max(months_covered, 1))

    essential_debit = sum(
        t.amount for t in debits if t.amount is not None and t.lifestyle_dim == _ESSENTIAL_DIM
    )
    discretionary_debit = sum(
        t.amount for t in debits if t.amount is not None and t.lifestyle_dim == _ASPIRATIONAL_DIM
    )
    essential_ratio = _safe_div(essential_debit, total_debit)
    discretionary_ratio = _safe_div(discretionary_debit, total_debit)

    dates = [t.txn_date for t in scoring if t.txn_date]
    days_span = max(((max(dates) - min(dates)).days + 1), 1) if dates else 1
    # Total daily burn rate, not just essential spend — someone who spends
    # almost entirely on discretionary categories (near-zero essential
    # spend) would otherwise divide by ~0 and get an absurdly inflated
    # "buffer", the opposite of what a thin-runway spender should show.
    avg_daily_spend = total_debit / days_span
    balances = [t.balance_val for t in scoring if t.balance_val is not None]
    avg_balance = _safe_div(sum(balances), len(balances)) if balances else 0.0
    balance_buffer_days = _safe_div(avg_balance, avg_daily_spend)

    cash_withdrawal_debit = sum(
        t.amount
        for t in debits
        if t.amount is not None and _CASH_WITHDRAWAL_RE.search(t.particulars or "")
    )
    cash_withdrawal_ratio = _safe_div(cash_withdrawal_debit, total_debit)

    merchant_debits = [t for t in debits if t.payee_type == "merchant"]
    resolved_merchant_debits = [
        t for t in merchant_debits if t.category is not None and t.category != "other"
    ]
    merchant_resolution_rate = _safe_div(len(resolved_merchant_debits), len(merchant_debits))

    bounce_count = sum(1 for t in usable if _BOUNCE_RE.search(t.particulars or ""))

    bnpl_debits = [t for t in debits if t.risk_flag == "bnpl_lending"]
    bnpl_merchant_count = len({(t.payee or "").strip().lower() for t in bnpl_debits if t.payee})
    bnpl_share = _safe_div(sum(t.amount for t in bnpl_debits if t.amount is not None), total_debit)

    gambling_debits = [t for t in debits if t.risk_flag == "gambling"]
    gambling_share = _safe_div(
        sum(t.amount for t in gambling_debits if t.amount is not None), total_debit
    )

    return FeatureVector(
        salary_detected=salary_detected,
        monthly_income=round(monthly_income, 2),
        foir=round(foir, 4),
        net_cashflow=round(net_cashflow, 2),
        balance_buffer_days=round(balance_buffer_days, 2),
        essential_ratio=round(essential_ratio, 4),
        discretionary_ratio=round(discretionary_ratio, 4),
        cash_withdrawal_ratio=round(cash_withdrawal_ratio, 4),
        merchant_resolution_rate=round(merchant_resolution_rate, 4),
        bounce_count=bounce_count,
        bnpl_merchant_count=bnpl_merchant_count,
        bnpl_share=round(bnpl_share, 4),
        gambling_share=round(gambling_share, 4),
        months_covered=months_covered,
        txn_count=len(usable),
    )
