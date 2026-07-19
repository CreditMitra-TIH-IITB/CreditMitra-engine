"""Issue #15 — persona ranking acceptance test.

docs/taxonomy.md: "Weights are expert-set, not fitted. Persona ranking (#15)
is the acceptance test." There's no labelled default data to validate
against, so this test suite is the substitute: each of the six synthetic
personas in tests/fixtures/personas/ was built to stress exactly one axis
of the scorer (docs/taxonomy.md's L-indices), and the acceptance criteria
are that (a) each classifies as its intended archetype and (b) the relative
ordering of scores and indices across personas makes real-world sense —
not exact point values, which were never fitted to anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.statements import CreditRiskReport, FeatureVector, LifestyleProfile, Transaction
from app.services.archetype import classify_archetype
from app.services.credit_scorer import score
from app.services.feature_engineering import build_features
from app.services.lifestyle_profile import build_profile

FIXTURES = Path(__file__).parent / "fixtures" / "personas"

PERSONAS = {
    "salaried_saver": "01_salaried_saver",
    "gig_hustler": "02_gig_hustler",
    "bnpl_heavy_spender": "03_bnpl_heavy_spender",
    "gambler": "04_gambler",
    "aspirational_overspender": "05_aspirational_overspender",
    "cash_reliant_informal": "06_cash_reliant_informal",
}

EXPECTED_ARCHETYPE = {
    "salaried_saver": "Salaried Saver",
    "gig_hustler": "Gig Hustler",
    "bnpl_heavy_spender": "BNPL-Heavy Spender",
    "gambler": "Gambler",
    "aspirational_overspender": "Aspirational Overspender",
    "cash_reliant_informal": "Cash-Reliant Informal",
}


def _load(persona: str) -> list[Transaction]:
    path = FIXTURES / f"{PERSONAS[persona]}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Transaction(**row) for row in data]


def _run(persona: str) -> tuple[FeatureVector, LifestyleProfile, str, CreditRiskReport]:
    txns = _load(persona)
    features = build_features(txns)
    profile = build_profile(txns, features)
    archetype = classify_archetype(features, profile)
    report = score(features, profile, archetype)
    return features, profile, archetype, report


@pytest.fixture(scope="module")
def results() -> dict[str, tuple[FeatureVector, LifestyleProfile, str, CreditRiskReport]]:
    return {persona: _run(persona) for persona in PERSONAS}


# ---------------------------------------------------------------------------
# Archetype classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("persona", list(PERSONAS))
def test_archetype_matches_intended_persona(results, persona):
    _, _, archetype, _ = results[persona]
    assert archetype == EXPECTED_ARCHETYPE[persona]


# ---------------------------------------------------------------------------
# Index-level sanity: each persona should be the (or among the) most
# extreme on the axis it was specifically built to stress.
# ---------------------------------------------------------------------------


def test_gambler_has_the_lowest_risk_appetite_index(results):
    l6_by_persona = {p: results[p][1].l6_risk_appetite for p in PERSONAS}
    assert min(l6_by_persona, key=l6_by_persona.get) == "gambler"


def test_bnpl_heavy_spender_has_the_lowest_leverage_index(results):
    l5_by_persona = {p: results[p][1].l5_leverage for p in PERSONAS}
    assert min(l5_by_persona, key=l5_by_persona.get) == "bnpl_heavy_spender"


def test_cash_reliant_informal_has_the_lowest_digital_maturity(results):
    l3_by_persona = {p: results[p][1].l3_digital_maturity for p in PERSONAS}
    assert min(l3_by_persona, key=l3_by_persona.get) == "cash_reliant_informal"


def test_aspirational_overspender_has_the_highest_aspirational_index(results):
    l2_by_persona = {p: results[p][1].l2_aspirational for p in PERSONAS}
    assert max(l2_by_persona, key=l2_by_persona.get) == "aspirational_overspender"


def test_salaried_saver_has_a_strong_commitment_index(results):
    """L4 is "the self-control proxy — most important" (docs/taxonomy.md).
    Salaried Saver is the only persona with sustained SIP + insurance +
    rent, so it should clear a high bar, not just edge out the others."""
    l4 = results["salaried_saver"][1].l4_commitment
    assert l4 >= 60


def test_gig_hustler_is_fully_digital(results):
    """The whole point of the archetype: no cash reliance, fully digital
    income and spend, despite no formal employer."""
    l3 = results["gig_hustler"][1].l3_digital_maturity
    assert l3 >= 80


# ---------------------------------------------------------------------------
# Overall score ranking
# ---------------------------------------------------------------------------


def test_gambler_scores_lowest_overall(results):
    """Active gambling/crypto exposure is the strongest negative signal
    this model tracks (docs/taxonomy.md L6 rationale) — it should pull the
    final score below every other persona, not just its own lifestyle
    block."""
    scores = {p: results[p][3].score for p in PERSONAS}
    assert min(scores, key=scores.get) == "gambler"


def test_disciplined_personas_outscore_leveraged_and_risky_ones(results):
    """The two "disciplined" archetypes (steady/sustained commitments, no
    leverage or risk exposure) should both outscore every archetype defined
    by a leverage or risk problem — the core ordering the weights exist to
    produce, even though exact point values are expert-set, not fitted."""
    disciplined = ["salaried_saver", "gig_hustler"]
    risky = ["bnpl_heavy_spender", "gambler"]
    min_disciplined = min(results[p][3].score for p in disciplined)
    max_risky = max(results[p][3].score for p in risky)
    assert min_disciplined > max_risky


def test_salaried_saver_scores_in_excellent_or_very_good_band(results):
    band = results["salaried_saver"][3].band
    assert band in {"Excellent", "Very Good"}


def test_gambler_does_not_score_in_the_top_two_bands(results):
    band = results["gambler"][3].band
    assert band not in {"Excellent", "Very Good"}


# ---------------------------------------------------------------------------
# Report shape sanity — every persona should produce a well-formed report,
# not just the ones exercised above.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("persona", list(PERSONAS))
def test_report_is_well_formed(results, persona):
    _, profile, archetype, report = results[persona]
    assert 300 <= report.score <= 900
    assert report.band in {"Poor", "Fair", "Good", "Very Good", "Excellent"}
    assert report.archetype == archetype
    assert report.lifestyle.archetype == archetype  # nested field kept in sync
    assert 1 <= len(report.factors) <= 6
    assert report.narrative  # never blank


# ---------------------------------------------------------------------------
# Fair lending — healthcare must never lower a score (docs/taxonomy.md)
# ---------------------------------------------------------------------------


def _base_healthcare_txns() -> list[Transaction]:
    from datetime import date

    return [
        Transaction(
            date="01-10-2025",
            particulars="SALARY",
            deposits="50000.00",
            withdrawals="",
            balance="50000.00",
            payee="Employer",
            payee_type="person",
            txn_date=date(2025, 10, 1),
            amount=50000.0,
            direction="credit",
            balance_val=50000.0,
        ),
        Transaction(
            date="05-10-2025",
            particulars="GROCERY",
            deposits="",
            withdrawals="3000.00",
            balance="47000.00",
            payee="BigBasket",
            payee_type="merchant",
            txn_date=date(2025, 10, 5),
            amount=3000.0,
            direction="debit",
            balance_val=47000.0,
            category="groceries",
            is_essential=True,
            lifestyle_dim="essential",
            recurring_type="adhoc",
        ),
        Transaction(
            date="10-11-2025",
            particulars="SALARY",
            deposits="50000.00",
            withdrawals="",
            balance="94000.00",
            payee="Employer",
            payee_type="person",
            txn_date=date(2025, 11, 1),
            amount=50000.0,
            direction="credit",
            balance_val=94000.0,
        ),
        Transaction(
            date="05-11-2025",
            particulars="GROCERY",
            deposits="",
            withdrawals="3000.00",
            balance="91000.00",
            payee="BigBasket",
            payee_type="merchant",
            txn_date=date(2025, 11, 5),
            amount=3000.0,
            direction="debit",
            balance_val=91000.0,
            category="groceries",
            is_essential=True,
            lifestyle_dim="essential",
            recurring_type="adhoc",
        ),
    ]


def _with_large_healthcare_bill() -> list[Transaction]:
    from datetime import date

    txns = _base_healthcare_txns()
    txns.append(
        Transaction(
            date="15-10-2025",
            particulars="HOSPITAL BILL",
            deposits="",
            withdrawals="80000.00",
            balance="-33000.00",
            payee="Apollo Hospital",
            payee_type="merchant",
            txn_date=date(2025, 10, 15),
            amount=80000.0,
            direction="debit",
            balance_val=-33000.0,
            category="healthcare",
            is_essential=True,
            lifestyle_dim="essential",
            recurring_type="adhoc",
        )
    )
    return txns


def test_healthcare_spend_does_not_change_the_score():
    """A large medical bill must not move the score at all — not lower it,
    not even indirectly through a diluted ratio. Full exclusion, not
    zero-weighting (see app/services/feature_engineering.py docstring).

    txn_count is deliberately the one field that DOES differ — it's a raw
    row-volume metric, not a scoring ratio, so the healthcare row still
    counts there (it's still a real transaction that happened); everything
    that actually feeds the score must be identical.
    """
    baseline_txns = _base_healthcare_txns()
    with_bill_txns = _with_large_healthcare_bill()

    baseline_features = build_features(baseline_txns)
    with_bill_features = build_features(with_bill_txns)
    assert with_bill_features.txn_count == baseline_features.txn_count + 1
    assert with_bill_features.model_copy(update={"txn_count": 0}) == baseline_features.model_copy(
        update={"txn_count": 0}
    )

    baseline_profile = build_profile(baseline_txns, baseline_features)
    with_bill_profile = build_profile(with_bill_txns, with_bill_features)
    assert baseline_profile == with_bill_profile

    baseline_report = score(baseline_features, baseline_profile, "Balanced")
    with_bill_report = score(with_bill_features, with_bill_profile, "Balanced")
    assert baseline_report.score == with_bill_report.score
