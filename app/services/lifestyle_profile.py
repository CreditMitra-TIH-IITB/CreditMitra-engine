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

from app.core import scoring_config as cfg
from app.schemas.statements import FeatureVector, LifestyleProfile, Transaction
from app.services.feature_engineering import is_rent

_RISK_FLAGS = {"gambling", "crypto"}

# A voluntary commitment is rent or a `commitment` row (SIP, insurance, loan
# EMI). Anything carrying a risk_flag is excluded on purpose: paying four BNPL
# apps on time every month is leverage, not character, and L5 already scores it.
# Counting it here would have let the most leveraged persona post a near-perfect
# commitment index.
_COMMITMENT_DIM = "commitment"

# Day-of-month spread at which a commitment stops looking scheduled at all.
_REGULARITY_SPREAD_DAYS = 8.0

# A commitment paid this reliably relative to income is "fully" sustained;
# research doc L4 uses 0.35 x income as the reference obligation load.
_COMMITMENT_INCOME_REFERENCE = 0.35


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


def _spending_persistence(debits: list[Transaction]) -> float:
    """Mean cosine similarity between consecutive months' category-spend
    vectors — Gladstone et al.'s "spending persistence". A person whose
    October looks like their November is running a stable life; one whose
    category mix churns month to month is not.

    Replaces the earlier stand-in ("did an essential row appear this month"),
    which saturated at 1.0 for anyone who bought groceries and so carried no
    information.
    """
    by_month: dict[tuple[int, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for t in debits:
        if t.txn_date and t.amount and t.category:
            by_month[_month_key(t.txn_date)][t.category] += t.amount

    months = sorted(by_month)
    if len(months) < 2:
        return 0.0

    similarities: list[float] = []
    for earlier, later in zip(months, months[1:], strict=False):
        a, b = by_month[earlier], by_month[later]
        shared = set(a) | set(b)
        dot = sum(a.get(c, 0.0) * b.get(c, 0.0) for c in shared)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a and norm_b:
            similarities.append(dot / (norm_a * norm_b))
    return sum(similarities) / len(similarities) if similarities else 0.0


def _l1_essential_stability(scoring: list[Transaction], features: FeatureVector) -> int:
    """Research doc L1: 100 x (0.6 x essential_share + 0.4 x persistence).

    The share weight leads, because *what* the money goes to is the primary
    claim; persistence modulates it. Both terms are raw ratios — the earlier
    version divided the share by 0.5 before capping, which let a profile with
    only 50% essential spend read as a perfect essential share.
    """
    debits = _debits(scoring)
    share = min(features.essential_ratio, 1.0)
    persistence = _spending_persistence(debits)
    return _clamp(100 * (0.6 * share + 0.4 * persistence))


def _l2_aspirational(features: FeatureVector) -> int:
    """Level of discretionary spend. Deliberately NOT penalised here — the
    "risky only with a thin buffer" nuance (docs/taxonomy.md) is a scorer
    concern (app/services/credit_scorer.py), not an index concern."""
    return _clamp(100 * min(features.discretionary_ratio / 0.5, 1.0))


def _l3_digital_maturity(features: FeatureVector) -> int:
    cash_penalty = min(features.cash_withdrawal_ratio / 0.3, 1.0)
    return _clamp(100 * (0.5 * (1 - cash_penalty) + 0.5 * features.merchant_resolution_rate))


def _is_voluntary_commitment(t: Transaction) -> bool:
    if t.risk_flag is not None:
        return False
    return is_rent(t) or t.lifestyle_dim == _COMMITMENT_DIM


def _l4_commitment(scoring: list[Transaction], features: FeatureVector) -> int:
    """The self-control proxy, and the heaviest-weighted index in the model.

    Research doc L4:

        L4 = 100 x min(1, SUM(recurring) / (0.35 x income)) x regularity_factor

    where a recurring commitment is a payee paid on a near-monthly cadence and
    `regularity_factor` falls as the day-of-month drifts. Three departures from
    a literal reading, each deliberate:

    1. Cadence is judged per *payee*, not per category, which is what makes
       "the same landlord every month" legible as one sustained obligation.
    2. A commitment only counts once it appears in >=3 distinct months. Two
       payments are a coincidence; three are a habit.
    3. The result is scaled by a reliability term. A commitment that bounces is
       not being sustained, whatever the calendar says — without it, the
       persona who bounced eight payments in six months scored a *perfect*
       commitment index off the rent line alone. This intentionally overlaps
       with the cash-flow block's bounce penalty: the two answer different
       questions (did you honour your obligations vs can you cover your
       outflows) and a bounce is real evidence for both.
    """
    debits = [t for t in _debits(scoring) if _is_voluntary_commitment(t)]
    if not debits:
        # No commitment rows at all is absence of evidence, not evidence of
        # failure, and the two must not score the same. Someone who pays rent
        # in cash is invisible here; someone whose rent ECS bounces every month
        # is visible and failing. Returning 0 for both handed the informal
        # cash earner the same penalty as the serial bouncer — precisely the
        # exclusion this project exists to argue against. Neutral means the
        # index contributes nothing and the score rests on what we can see.
        return cfg.neutral_for("l4_commitment")

    by_payee: dict[str, list[Transaction]] = defaultdict(list)
    for t in debits:
        # Rent shares a bucket regardless of how the landlord's name parsed.
        key = "__rent__" if is_rent(t) else (t.payee or t.category or "").strip().lower()
        if key:
            by_payee[key].append(t)

    monthly_total = 0.0
    day_spreads: list[float] = []
    for txns in by_payee.values():
        months = {_month_key(t.txn_date) for t in txns if t.txn_date}
        if len(months) < 3:
            continue
        amounts = [t.amount for t in txns if t.amount is not None]
        if not amounts:
            continue
        # Per-month contribution, so a commitment paid in 4 of 6 months counts
        # for what it actually is rather than its full monthly sticker value.
        monthly_total += sum(amounts) / max(features.months_covered, 1)

        days = [t.txn_date.day for t in txns if t.txn_date]
        if len(days) >= 2:
            mean_day = sum(days) / len(days)
            variance = sum((d - mean_day) ** 2 for d in days) / len(days)
            day_spreads.append(math.sqrt(variance))

    if monthly_total <= 0:
        return 0

    reference = _COMMITMENT_INCOME_REFERENCE * features.monthly_income
    magnitude = min(monthly_total / reference, 1.0) if reference > 0 else 0.0

    spread = sum(day_spreads) / len(day_spreads) if day_spreads else 0.0
    regularity = 1 - min(spread / _REGULARITY_SPREAD_DAYS, 1.0)

    reliability = 1 - min(_safe_div(features.bounce_count, max(features.months_covered, 1)), 1.0)

    return _clamp(100 * magnitude * regularity * reliability)


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
