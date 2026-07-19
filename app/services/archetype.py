"""Archetype classification (Track B).

ISSUE #12. classify_archetype(features, lifestyle) -> str.

Rule-based, priority-ordered — checked most-severe-and-specific first, so a
profile that could match more than one archetype (e.g. a gambler who also
uses BNPL) lands on the more actionable label rather than whichever rule
happens to run last. Thresholds were tuned against the six synthetic
personas in tests/fixtures/personas/ (see Issue #15's acceptance test) —
each was built to stress exactly one axis, so getting all six to classify
correctly is the calibration signal in the absence of labelled real data
(docs/taxonomy.md: "Persona ranking is the acceptance test").

Pure function over FeatureVector + LifestyleProfile — no raw transactions,
so it stays cheap to call from the scorer and easy to unit test.
"""

from __future__ import annotations

from app.schemas.statements import FeatureVector, LifestyleProfile

BALANCED = "Balanced"
GAMBLER = "Gambler"
BNPL_HEAVY_SPENDER = "BNPL-Heavy Spender"
CASH_RELIANT_INFORMAL = "Cash-Reliant Informal"
GIG_HUSTLER = "Gig Hustler"
ASPIRATIONAL_OVERSPENDER = "Aspirational Overspender"
SALARIED_SAVER = "Salaried Saver"


def classify_archetype(features: FeatureVector, lifestyle: LifestyleProfile) -> str:
    """Never raises — falls through to BALANCED if nothing distinctive matches."""

    # 1. Gambler — L6 is the inverse risk-appetite index; low means high
    # gambling/crypto exposure. Checked first: active risk-taking is the
    # highest-severity signal regardless of what else is going on.
    if lifestyle.l6_risk_appetite < 40:
        return GAMBLER

    # 2. BNPL-Heavy Spender — L5 inverse leverage index bottomed out by
    # buy-now-pay-later usage. Hidden debt the person may not think of as
    # debt (docs/taxonomy.md).
    if lifestyle.l5_leverage < 40:
        return BNPL_HEAVY_SPENDER

    # 3. Cash-Reliant Informal — low digital maturity: heavy ATM reliance,
    # little resolvable digital spend. Checked before Gig Hustler because
    # both can show salary_detected=False; digital maturity is what tells
    # them apart (a gig worker is fully digital, an informal-cash earner
    # isn't).
    if lifestyle.l3_digital_maturity < 60:
        return CASH_RELIANT_INFORMAL

    # 4. Gig Hustler — no formal employer salary, but real recurring
    # income exists (gig-platform payouts, reinterpreted in
    # feature_engineering.py), and the person transacts entirely digitally.
    if (
        not features.salary_detected
        and features.monthly_income > 0
        and lifestyle.l3_digital_maturity >= 70
    ):
        return GIG_HUSTLER

    # 5. Aspirational Overspender — docs/taxonomy.md: L2 (aspirational
    # spend level) is "neutral alone; risky only w/ low buffer". High
    # discretionary spend only earns this label when the balance runway is
    # actually thin.
    if lifestyle.l2_aspirational >= 70 and features.balance_buffer_days < 30:
        return ASPIRATIONAL_OVERSPENDER

    # 6. Salaried Saver — detected formal income, sustained voluntary
    # commitments, low leverage/risk. The profile the traditional scoring
    # model already rewards.
    if features.salary_detected and lifestyle.l4_commitment >= 50 and lifestyle.l5_leverage >= 70:
        return SALARIED_SAVER

    return BALANCED
