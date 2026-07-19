"""Lifestyle indices — the core IP (Track B).

ISSUE #11. build_profile(transactions, features) -> LifestyleProfile.

Six 0-100 indices (docs/taxonomy.md §Scorecard weights), plus behavioural
texture metrics from Gladstone et al. (EPJ Data Science, 2021) and the
Goh-Barabasi burstiness measure. Archetype labelling itself lives in
app/services/archetype.py (Issue #12) — this module only computes the
numbers an archetype rule (or a human) would read.

FAIR LENDING: healthcare transactions are excluded from every index here,
same as app/services/feature_engineering.py — see that module's docstring
for why (full exclusion, not zero-weighting).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date

from app.schemas.statements import FeatureVector, LifestyleProfile, Transaction

_COMMITTED_CATEGORIES = {"investments", "insurance", "rent", "loan_emi"}
_RISK_FLAGS = {"gambling", "crypto"}


def _month_key(d: date) -> tuple[int, int]:
    return (d.year, d.month)


def _scoring_relevant(transactions: list[Transaction]) -> list[Transaction]:
    return [
        t
        for t in transactions
        if t.txn_date is not None and t.amount is not None and t.category != "healthcare"
    ]


def _debits(transactions: list[Transaction]) -> list[Transaction]:
    return [t for t in transactions if t.direction == "debit"]


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> int:
    return int(round(max(lo, min(hi, value))))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


# ---------------------------------------------------------------------------
# L-indices
# ---------------------------------------------------------------------------


def _l1_essential_stability(scoring: list[Transaction], features: FeatureVector) -> int:
    """Persistence (essentials paid most months) + a healthy essential share."""
    debits = _debits(scoring)
    months = {_month_key(t.txn_date) for t in debits if t.txn_date}
    months_with_essential = {
        _month_key(t.txn_date) for t in debits if t.lifestyle_dim == "essential" and t.txn_date
    }
    persistence = _safe_div(len(months_with_essential), max(len(months), 1))
    share = min(features.essential_ratio / 0.5, 1.0)
    return _clamp(100 * (0.6 * persistence + 0.4 * share))


def _l2_aspirational(features: FeatureVector) -> int:
    """Level of discretionary spend. Deliberately NOT penalised here — the
    "risky only with a thin buffer" nuance (docs/taxonomy.md) is a scorer
    concern (app/services/credit_scorer.py), not an index concern."""
    return _clamp(100 * min(features.discretionary_ratio / 0.5, 1.0))


def _l3_digital_maturity(features: FeatureVector) -> int:
    cash_penalty = min(features.cash_withdrawal_ratio / 0.3, 1.0)
    return _clamp(100 * (0.5 * (1 - cash_penalty) + 0.5 * features.merchant_resolution_rate))


def _l4_commitment(scoring: list[Transaction], features: FeatureVector) -> int:
    """The self-control proxy. Rewards SUSTAINED (>=2 distinct months)
    voluntary commitments — SIP, insurance, rent, loan EMI — over a merely
    healthy FOIR, because persistence across months is the actual signal
    (docs/taxonomy.md: "rewards voluntary sustained obligations")."""
    debits = _debits(scoring)
    months_by_category: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for t in debits:
        if t.category in _COMMITTED_CATEGORIES and t.txn_date:
            months_by_category[t.category].add(_month_key(t.txn_date))
    sustained_types = sum(1 for months in months_by_category.values() if len(months) >= 2)

    foir_band = 1 - min(abs(features.foir - 0.25) / 0.35, 1.0)  # healthiest near foir=0.25
    return _clamp(min(sustained_types, 3) * 30 + foir_band * 10)


def _l5_leverage(features: FeatureVector) -> int:
    """Inverse: high = LOW leverage (good)."""
    share_penalty = min(features.bnpl_share / 0.15, 1.0) * 70
    count_penalty = min(features.bnpl_merchant_count / 4, 1.0) * 30
    return _clamp(100 - share_penalty - count_penalty)


def _l6_risk_appetite(scoring: list[Transaction]) -> int:
    """Inverse: high = LOW risk appetite (good). Gambling + crypto share of
    total debit spend — crypto isn't in FeatureVector (schema is frozen,
    Issue #1), so it's computed directly from transactions here."""
    debits = _debits(scoring)
    total_debit = sum(t.amount for t in debits if t.amount is not None)
    risk_debit = sum(
        t.amount for t in debits if t.amount is not None and t.risk_flag in _RISK_FLAGS
    )
    risk_share = _safe_div(risk_debit, total_debit)
    return _clamp(100 - min(risk_share / 0.10, 1.0) * 100)


# ---------------------------------------------------------------------------
# Behavioural texture (Gladstone et al., EPJ Data Science 2021)
# ---------------------------------------------------------------------------


def _category_diversity(scoring: list[Transaction]) -> float:
    """Normalized Shannon entropy of the category distribution over debits
    with a resolved category. 0 = single category, 1 = maximally spread."""
    debits = [t for t in _debits(scoring) if t.category and t.category != "other"]
    if len(debits) < 2:
        return 0.0
    counts = Counter(t.category for t in debits)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    return round(_safe_div(entropy, max_entropy), 4)


def _merchant_loyalty(scoring: list[Transaction]) -> float:
    """Repeat-merchant txn share: of all merchant debits, how many are NOT
    the first visit to that merchant."""
    debits = [t for t in _debits(scoring) if t.payee_type == "merchant" and t.payee]
    if not debits:
        return 0.0
    names = [t.payee.strip().lower() for t in debits]
    distinct = len(set(names))
    return round(_safe_div(len(names) - distinct, len(names)), 4)


def _burstiness(scoring: list[Transaction]) -> float:
    """Goh-Barabasi B = (sigma - mu) / (sigma + mu) over inter-transaction
    intervals (days between consecutive transactions, sorted by date).
    B > 0: bursty/clustered activity. B < 0: regular/periodic. Needs >=3
    transactions (2+ intervals) to mean anything; else 0.0 (neutral)."""
    dates = sorted(t.txn_date for t in scoring if t.txn_date)
    if len(dates) < 3:
        return 0.0
    intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    n = len(intervals)
    mu = sum(intervals) / n
    variance = sum((x - mu) ** 2 for x in intervals) / n
    sigma = math.sqrt(variance)
    if sigma + mu == 0:
        return 0.0
    return round((sigma - mu) / (sigma + mu), 4)


def _category_turnover(scoring: list[Transaction], months_covered: int) -> float:
    """Distinct categories seen, averaged per month covered — a coarse
    proxy for "how many different kinds of spending appear per month",
    not a true first-appearance count (that needs ordered per-month
    category sets, which is more machinery than this signal is worth)."""
    debits = [t for t in _debits(scoring) if t.category and t.category != "other"]
    distinct_categories = len({t.category for t in debits})
    return round(_safe_div(distinct_categories, max(months_covered, 1)), 4)


def build_profile(transactions: list[Transaction], features: FeatureVector) -> LifestyleProfile:
    """Compute the six L-indices and behavioural texture metrics.

    `archetype` is left as the schema default ("unknown") here — set it via
    app/services/archetype.py's classify_archetype(), composed by the
    caller (app/services/credit_scorer.py / app/services/extraction.py).
    Never raises — sparse input just produces low/neutral index values.
    """
    scoring = _scoring_relevant(transactions)

    return LifestyleProfile(
        l1_essential_stability=_l1_essential_stability(scoring, features),
        l2_aspirational=_l2_aspirational(features),
        l3_digital_maturity=_l3_digital_maturity(features),
        l4_commitment=_l4_commitment(scoring, features),
        l5_leverage=_l5_leverage(features),
        l6_risk_appetite=_l6_risk_appetite(scoring),
        category_diversity=_category_diversity(scoring),
        merchant_loyalty=_merchant_loyalty(scoring),
        burstiness=_burstiness(scoring),
        category_turnover=_category_turnover(scoring, features.months_covered),
    )
