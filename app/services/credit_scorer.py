"""Final score assembly (Track B). ISSUE #13.

score(features, lifestyle, archetype) -> CreditRiskReport. Combines the
lifestyle block (six L-indices, docs/taxonomy.md's weighted formula) and a
cash-flow block (income regularity, FOIR, balance buffer, bounces) into a
300-900 score, band, human-readable factors, and a one-line narrative.

All weights/constants live in app/core/scoring_config.py — nowhere else
(docs/taxonomy.md is explicit about this).

FAIR LENDING: healthcare is excluded upstream, in feature_engineering.py and
lifestyle_profile.py — every number this module reads already has it
removed. This module adds no further exclusion logic itself; it just must
never reach into raw transactions and re-introduce it.
"""

from __future__ import annotations

from app.core import scoring_config as cfg
from app.schemas.statements import CreditRiskReport, FeatureVector, LifestyleProfile, ScoreFactor

_LIFESTYLE_LABELS: dict[str, str] = {
    "l4_commitment": "Commitment Index",
    "l1_essential_stability": "Essential Stability",
    "l5_leverage": "Leverage Index",
    "l6_risk_appetite": "Risk Appetite Index",
    "l3_digital_maturity": "Digital Maturity",
    "l2_aspirational": "Aspirational Index",
}

_NARRATIVES: dict[str, str] = {
    "Salaried Saver": (
        "Steady salary, sustained SIP/insurance/rent, low leverage and risk — "
        "the profile a traditional score already rewards well."
    ),
    "Gig Hustler": (
        "Income comes from gig-platform payouts rather than a single employer, "
        "but spending stays disciplined and fully digital — the profile a "
        "salary-only score would miss entirely."
    ),
    "BNPL-Heavy Spender": (
        "Multiple buy-now-pay-later apps carry a meaningful share of spend — "
        "debt that doesn't show up as a loan but behaves like one."
    ),
    "Gambler": (
        "A significant share of spend goes to gambling and/or crypto platforms, "
        "the strongest risk-appetite signal this model tracks."
    ),
    "Aspirational Overspender": (
        "Discretionary spend is high relative to a thin balance buffer — "
        "lifestyle spend that isn't backed by much of a cushion."
    ),
    "Cash-Reliant Informal": (
        "Heavy reliance on cash withdrawals over digital payments leaves "
        "little resolvable transaction history to score against."
    ),
    "Balanced": "No single spending pattern dominates this statement.",
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _lifestyle_block(lifestyle: LifestyleProfile) -> tuple[float, list[ScoreFactor]]:
    """docs/taxonomy.md: "Σ weightᵢ × (Lᵢ − 50)/50 × max_pointsᵢ". max_points
    already IS weight_i × 300 (the column sums to exactly 300 — the whole
    block's budget) — it's not a separate factor to multiply in again. Using
    `weight * (L-50)/50 * max_points` double-applies the weight and crushes
    every index's swing to ~30% of its intended size."""
    factors: list[ScoreFactor] = []
    total = 0.0
    for field, (_weight, max_points) in cfg.LIFESTYLE_WEIGHTS.items():
        value: int = getattr(lifestyle, field)
        impact = (value - 50) / 50 * max_points
        total += impact
        if abs(impact) >= 3:
            label = _LIFESTYLE_LABELS[field]
            direction = "supports" if impact >= 0 else "weighs on"
            factors.append(
                ScoreFactor(
                    factor=f"{label} {value} — {direction} the score",
                    impact=round(impact),
                    block="lifestyle",
                )
            )
    return total, factors


def _cashflow_block(features: FeatureVector) -> tuple[float, list[ScoreFactor]]:
    factors: list[ScoreFactor] = []
    total = 0.0

    if features.salary_detected:
        pts = cfg.INCOME_SALARY_POINTS
        factors.append(
            ScoreFactor(
                factor=f"Regular salary detected (~Rs {features.monthly_income:,.0f}/mo)",
                impact=pts,
                block="cashflow",
            )
        )
        total += pts
    elif features.monthly_income > 0:
        pts = cfg.INCOME_GIG_POINTS
        factors.append(
            ScoreFactor(
                factor=(
                    f"Recurring income detected without a single employer "
                    f"(~Rs {features.monthly_income:,.0f}/mo)"
                ),
                impact=pts,
                block="cashflow",
            )
        )
        total += pts
    else:
        factors.append(
            ScoreFactor(
                factor="No regular income pattern detected in this statement",
                impact=0,
                block="cashflow",
            )
        )

    foir_closeness = 1 - abs(features.foir - cfg.FOIR_HEALTHY_TARGET) / cfg.FOIR_TOLERANCE
    foir_pts = round(
        _clamp(cfg.FOIR_MAX_POINTS * foir_closeness, cfg.FOIR_MIN_POINTS, cfg.FOIR_MAX_POINTS)
    )
    total += foir_pts
    if abs(foir_pts) >= 3:
        factors.append(
            ScoreFactor(
                factor=f"Fixed obligations are {features.foir:.0%} of income (FOIR)",
                impact=foir_pts,
                block="cashflow",
            )
        )

    buffer_pts = round(
        _clamp(
            (features.balance_buffer_days - cfg.BUFFER_NEUTRAL_DAYS) * cfg.BUFFER_POINTS_PER_DAY,
            cfg.BUFFER_MIN_POINTS,
            cfg.BUFFER_MAX_POINTS,
        )
    )
    total += buffer_pts
    if abs(buffer_pts) >= 3:
        factors.append(
            ScoreFactor(
                factor=f"Average balance covers ~{features.balance_buffer_days:.0f} days of spend",
                impact=buffer_pts,
                block="cashflow",
            )
        )

    if features.bounce_count > 0:
        bounce_pts = round(
            _clamp(-cfg.BOUNCE_PENALTY_PER_EVENT * features.bounce_count, cfg.BOUNCE_MIN_POINTS, 0)
        )
        total += bounce_pts
        factors.append(
            ScoreFactor(
                factor=f"{features.bounce_count} bounced/returned payment(s) detected",
                impact=bounce_pts,
                block="cashflow",
            )
        )

    sign_pts = cfg.CASHFLOW_SIGN_POINTS if features.net_cashflow > 0 else -cfg.CASHFLOW_SIGN_POINTS
    total += sign_pts
    factors.append(
        ScoreFactor(
            factor="Positive net cashflow across the statement period"
            if features.net_cashflow > 0
            else "Negative net cashflow across the statement period",
            impact=sign_pts,
            block="cashflow",
        )
    )

    total = _clamp(total, -cfg.CASHFLOW_BLOCK_CAP, cfg.CASHFLOW_BLOCK_CAP)
    return total, factors


def score(features: FeatureVector, lifestyle: LifestyleProfile, archetype: str) -> CreditRiskReport:
    """Never raises — a sparse FeatureVector/LifestyleProfile (all zeros)
    still produces a valid report, just one that lands near BASE_SCORE."""
    # Keep the nested LifestyleProfile's own archetype field in sync with the
    # report-level one — build_profile() leaves it at the schema default
    # ("unknown") since archetype classification is this module's job, not
    # lifestyle_profile.py's.
    lifestyle.archetype = archetype

    lifestyle_points, lifestyle_factors = _lifestyle_block(lifestyle)
    cashflow_points, cashflow_factors = _cashflow_block(features)

    raw_score = cfg.BASE_SCORE + lifestyle_points + cashflow_points
    final_score = int(round(_clamp(raw_score, cfg.SCORE_MIN, cfg.SCORE_MAX)))
    band = cfg.band_for(final_score)

    factors = sorted(lifestyle_factors + cashflow_factors, key=lambda f: -abs(f.impact))[:6]

    return CreditRiskReport(
        score=final_score,
        band=band,
        archetype=archetype,
        factors=factors,
        lifestyle=lifestyle,
        features=features,
        narrative=_NARRATIVES.get(archetype, _NARRATIVES["Balanced"]),
    )
