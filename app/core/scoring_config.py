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
# Lowered from 600 so the strongest realistic profile lands just under the
# ceiling instead of being clamped there. At 600 the top two personas both
# pinned at exactly 900 and became indistinguishable — the model had no
# headroom left to say one was better than the other.
BASE_SCORE = 570

# ---------------------------------------------------------------------------
# Lifestyle block (~±300): docs/taxonomy.md "Scorecard weights" table.
# Σ weight_i × (L_i − 50)/50 × max_points_i
# ---------------------------------------------------------------------------

# field name on LifestyleProfile -> points swung at index 100 (0 points at 50).
#
# L2 is deliberately absent. Every index here is scored "higher is better", and
# L2 does not work that way: it measures how much of the budget goes to
# discretionary categories, which the research doc calls a neutral descriptor
# that "becomes risk only when combined with low buffer/high FOIR". Scoring it
# on the same ramp meant a persona spending 68% of income on food delivery and
# shopping earned points for it and landed in "Excellent". It is now a
# conditional penalty instead — see ASPIRATIONAL_STRAIN_* below.
# field -> (neutral, max_gain, max_penalty)
#
# Each index gets its own neutral anchor rather than a shared 50. The inverse
# indices are the reason: L5 and L6 read 100 for anybody who simply has no BNPL
# app and places no bets, which is most people. Anchoring them at 50 handed
# every unremarkable profile ~+90 for the absence of two vices, which is what
# pushed a persona with eight bounced payments toward the top of the range.
#
# The asymmetry is deliberate and matches how an underwriter reads these:
# having no gambling exposure is expected and earns a little; heavy exposure is
# disqualifying and costs a lot. L4 and L1 stay symmetric at 50, because
# sustained commitments and essential stability are genuine positive evidence,
# not merely the absence of something bad.
LIFESTYLE_WEIGHTS: dict[str, tuple[int, int, int]] = {
    "l4_commitment": (50, 72, 72),
    "l1_essential_stability": (50, 52, 52),
    "l5_leverage": (85, 20, 90),
    "l6_risk_appetite": (85, 15, 90),
    "l3_digital_maturity": (60, 24, 50),
}

# L2 as the doc actually specifies it: high discretionary spend is only a
# problem when there is no buffer behind it. Both conditions have to hold, and
# the penalty scales with how far past each threshold the profile sits.
ASPIRATIONAL_STRAIN_INDEX = 60
ASPIRATIONAL_STRAIN_BUFFER_DAYS = 30
ASPIRATIONAL_STRAIN_MAX_PENALTY = 90


def neutral_for(field: str) -> int:
    """The index value that scores exactly zero points — i.e. "no opinion".
    Exposed so an index can return it when the statement carries no evidence
    either way, rather than bottoming out at 0 and being read as bad news."""
    return LIFESTYLE_WEIGHTS[field][0]


# ---------------------------------------------------------------------------
# Cash-flow block (~±300): FOIR, income regularity, balance buffer, bounces.
# ---------------------------------------------------------------------------

INCOME_SALARY_POINTS = 60
INCOME_GIG_POINTS = 42

# FOIR rewards a HEALTHY MIDDLE, not "lower is always better" — foir=0 means
# either "debt-free" or "no formal commitments at all" and those are not the
# same thing. Zero commitments is exactly what l4_commitment already scores
# as the worst case; if FOIR separately maxed out its points at foir=0, a
# profile with no SIP/insurance/rent/EMI would get rewarded twice for the
# same underlying fact by two different mechanisms with opposite intent.
FOIR_HEALTHY_TARGET = 0.25
FOIR_TOLERANCE = 0.35
FOIR_MIN_POINTS = -60
FOIR_MAX_POINTS = 45

BUFFER_NEUTRAL_DAYS = 15
BUFFER_POINTS_PER_DAY = 2
BUFFER_MIN_POINTS = -40
BUFFER_MAX_POINTS = 45

# A bounce is the single most direct evidence of inability to meet an
# obligation, so the penalty has to keep biting well past the second one. The
# old floor of -80 meant two bounces and eight bounces cost exactly the same,
# and a persona with eight bounced payments and negative cash flow still landed
# in "Very Good".
BOUNCE_PENALTY_PER_EVENT = 35
BOUNCE_MIN_POINTS = -180

CASHFLOW_SIGN_POINTS = 15

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
