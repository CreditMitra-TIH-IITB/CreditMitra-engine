"""Scoring weights and bands (Track B). ISSUE #13.

The ONLY place these numbers live — docs/taxonomy.md is explicit that this
must stay centralized, not duplicated across build_profile/credit_scorer.

Weights are expert-set, not fitted (no labelled default data yet). Sanity
is validated by the persona ranking acceptance test (Issue #15), not by a
training loop — see docs/taxonomy.md's "Scorecard weights" section.
"""

from __future__ import annotations

SCORE_MIN = 300
SCORE_MAX = 900
BASE_SCORE = 600

# ---------------------------------------------------------------------------
# Lifestyle block (~±300): docs/taxonomy.md "Scorecard weights" table.
# Σ weight_i × (L_i − 50)/50 × max_points_i
# ---------------------------------------------------------------------------

LIFESTYLE_WEIGHTS: dict[str, tuple[float, int]] = {
    # field name on LifestyleProfile: (weight, max_points)
    "l4_commitment": (0.30, 90),
    "l1_essential_stability": (0.20, 60),
    "l5_leverage": (0.20, 60),
    "l6_risk_appetite": (0.15, 45),
    "l3_digital_maturity": (0.10, 30),
    "l2_aspirational": (0.05, 15),
}

# ---------------------------------------------------------------------------
# Cash-flow block (~±300): FOIR, income regularity, balance buffer, bounces.
# ---------------------------------------------------------------------------

INCOME_SALARY_POINTS = 80
INCOME_GIG_POINTS = 50

# FOIR rewards a HEALTHY MIDDLE, not "lower is always better" — foir=0 means
# either "debt-free" or "no formal commitments at all" and those are not the
# same thing. Zero commitments is exactly what l4_commitment already scores
# as the worst case; if FOIR separately maxed out its points at foir=0, a
# profile with no SIP/insurance/rent/EMI would get rewarded twice for the
# same underlying fact by two different mechanisms with opposite intent.
FOIR_HEALTHY_TARGET = 0.25
FOIR_TOLERANCE = 0.35
FOIR_MIN_POINTS = -60
FOIR_MAX_POINTS = 60

BUFFER_NEUTRAL_DAYS = 15
BUFFER_POINTS_PER_DAY = 2
BUFFER_MIN_POINTS = -40
BUFFER_MAX_POINTS = 60

BOUNCE_PENALTY_PER_EVENT = 40
BOUNCE_MIN_POINTS = -80

CASHFLOW_SIGN_POINTS = 20

CASHFLOW_BLOCK_CAP = 300

# ---------------------------------------------------------------------------
# Bands — <580 Poor · 580-669 Fair · 670-739 Good · 740-799 Very Good · 800+
# ---------------------------------------------------------------------------

_BAND_THRESHOLDS: list[tuple[int, str]] = [
    (580, "Poor"),
    (670, "Fair"),
    (740, "Good"),
    (800, "Very Good"),
]
BAND_TOP = "Excellent"


def band_for(score: int) -> str:
    for threshold, label in _BAND_THRESHOLDS:
        if score < threshold:
            return label
    return BAND_TOP
