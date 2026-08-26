"""Issue #15 — persona ranking acceptance test.

docs/taxonomy.md: "Weights are expert-set, not fitted. Persona ranking (#15)
is the acceptance test." There's no labelled default data to validate
against, so this suite is the substitute: each of the eight synthetic personas
in tests/fixtures/personas/ was built to stress one axis of the scorer, and
the acceptance criteria are that (a) each classifies as its intended archetype
and (b) the relative ordering of scores and indices makes real-world sense —
not exact point values, which were never fitted to anything.

CALIBRATION GUARDS. Ordering tests alone turned out to be far too weak. The
previous two-month fixtures passed every one of them while the model was badly
mis-calibrated underneath: four of six indices were effectively constant, the
cash-flow block was positive for every persona regardless of behaviour, the top
persona was clamped at the ceiling, and a statement with eight bounced payments
scored "Very Good". Every ordering assertion still held, because each persona
had exactly one hand-placed dip.

The tests under "Calibration" exist to catch that class of failure: they assert
the model actually *discriminates* — that indices vary, that the cash-flow block
can go negative, that the range is used, that nothing pins to a bound. Those are
the properties an ordering test cannot see.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core import scoring_config as cfg
from app.schemas.statements import CreditRiskReport, FeatureVector, LifestyleProfile, Transaction
from app.services.archetype import classify_archetype
from app.services.credit_scorer import _cashflow_block, _lifestyle_block, score
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
    "frequent_bouncer": "07_frequent_bouncer",
    "balanced_salaried": "08_balanced_salaried",
}

EXPECTED_ARCHETYPE = {
    "salaried_saver": "Salaried Saver",
    "gig_hustler": "Gig Hustler",
    "bnpl_heavy_spender": "BNPL-Heavy Spender",
    "gambler": "Gambler",
    "aspirational_overspender": "Aspirational Overspender",
    "cash_reliant_informal": "Cash-Reliant Informal",
    "frequent_bouncer": "Overextended",
    "balanced_salaried": "Balanced",
}

L_INDEX_FIELDS = (
    "l1_essential_stability",
    "l2_aspirational",
    "l3_digital_maturity",
    "l4_commitment",
    "l5_leverage",
    "l6_risk_appetite",
)


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


def _scores(results) -> dict[str, int]:
    return {p: results[p][3].score for p in PERSONAS}


# ---------------------------------------------------------------------------
# Fixture integrity — the properties the calibration guards depend on.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("persona", list(PERSONAS))
def test_persona_spans_enough_months_to_judge_recurrence(results, persona):
    """L1 persistence and L4 cadence both need several months. At two months
    neither can move, which is what made the original fixtures useless for
    calibration."""
    features, _, _, _ = results[persona]
    assert features.months_covered >= 6


def test_the_set_contains_bounced_payments_and_losing_months(results):
    """Without either, the cash-flow block has nothing to push against and
    silently degenerates into a constant."""
    assert any(results[p][0].bounce_count > 0 for p in PERSONAS)
    assert any(results[p][0].net_cashflow < 0 for p in PERSONAS)


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
    l6 = {p: results[p][1].l6_risk_appetite for p in PERSONAS}
    assert min(l6, key=l6.get) == "gambler"


def test_bnpl_heavy_spender_has_the_lowest_leverage_index(results):
    l5 = {p: results[p][1].l5_leverage for p in PERSONAS}
    assert min(l5, key=l5.get) == "bnpl_heavy_spender"


def test_cash_reliant_informal_has_the_lowest_digital_maturity(results):
    l3 = {p: results[p][1].l3_digital_maturity for p in PERSONAS}
    assert min(l3, key=l3.get) == "cash_reliant_informal"


def test_aspirational_overspender_has_the_highest_aspirational_index(results):
    l2 = {p: results[p][1].l2_aspirational for p in PERSONAS}
    assert max(l2, key=l2.get) == "aspirational_overspender"


def test_salaried_saver_has_the_strongest_commitment_index(results):
    """L4 is the heaviest-weighted index. Salaried Saver is the only persona
    paying rent, insurance and a SIP on the same day every month."""
    l4 = {p: results[p][1].l4_commitment for p in PERSONAS}
    assert max(l4, key=l4.get) == "salaried_saver"
    assert l4["salaried_saver"] >= 90


def test_gig_hustler_is_fully_digital(results):
    l3 = results["gig_hustler"][1].l3_digital_maturity
    assert l3 >= 80


def test_rent_paid_to_a_landlord_counts_as_a_commitment(results):
    """Rent goes to a person, so no merchant tier resolves it and its category
    stays None. It has to be recognised from the narration — without that, the
    single largest recurring obligation most renters have is invisible to L4,
    to FOIR and to the essential share."""
    features, profile, _, _ = results["aspirational_overspender"]
    assert profile.l4_commitment > 0, "rent should register as a commitment"
    assert features.foir > 0, "rent should count toward fixed obligations"
    assert features.essential_ratio > 0.1, "rent should count as essential spend"


def test_bouncing_commitments_are_not_credited_as_commitment(results):
    """A commitment that bounces is not being sustained. The bouncer pays rent
    and a Bajaj EMI on a fixed schedule every month, so on cadence alone it
    would post a near-perfect commitment index."""
    assert results["frequent_bouncer"][1].l4_commitment < 20


def test_absent_commitments_score_neutral_not_worst(results):
    """Absence of evidence is not evidence of failure. Someone paying rent in
    cash is invisible to L4; someone whose payments bounce is visible and
    failing. Scoring both at zero penalised the informal earner for being
    unbanked, which is the exclusion this project argues against."""
    invisible = results["cash_reliant_informal"][1].l4_commitment
    failing = results["frequent_bouncer"][1].l4_commitment
    assert invisible == cfg.neutral_for("l4_commitment")
    assert invisible > failing


# ---------------------------------------------------------------------------
# Overall score ranking
# ---------------------------------------------------------------------------


def test_gambler_scores_lowest_overall(results):
    scores = _scores(results)
    assert min(scores, key=scores.get) == "gambler"


def test_disciplined_personas_outscore_leveraged_and_risky_ones(results):
    disciplined = ["salaried_saver", "gig_hustler", "balanced_salaried"]
    risky = ["bnpl_heavy_spender", "gambler", "frequent_bouncer"]
    assert min(results[p][3].score for p in disciplined) > max(results[p][3].score for p in risky)


def test_salaried_saver_scores_in_excellent_or_very_good_band(results):
    assert results["salaried_saver"][3].band in {"Excellent", "Very Good"}


def test_gambler_does_not_score_in_the_top_two_bands(results):
    assert results["gambler"][3].band not in {"Excellent", "Very Good"}


def test_serial_bouncer_lands_in_the_bottom_band(results):
    """The headline calibration regression: eight bounced payments and negative
    net cash flow used to produce 755 / "Very Good", because the bounce penalty
    saturated after two events and the lifestyle indices saw nothing unusual."""
    report = results["frequent_bouncer"][3]
    assert report.band == "Poor"
    assert report.score < 500


def test_gig_hustler_is_not_punished_for_lacking_an_employer(results):
    """The project's central claim: platform income plus disciplined spending
    should score like a good borrower, not like an unemployed one."""
    report = results["gig_hustler"][3]
    assert report.band in {"Excellent", "Very Good"}
    assert not results["gig_hustler"][0].salary_detected


# ---------------------------------------------------------------------------
# Calibration — that the model discriminates, not merely orders correctly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", L_INDEX_FIELDS)
def test_every_lifestyle_index_varies_across_personas(results, field):
    """An index that reads the same for everyone contributes nothing but a
    constant offset. L4 — the heaviest-weighted of the six — previously read
    ~3/100 for five of six personas while every ordering test still passed."""
    values = [getattr(results[p][1], field) for p in PERSONAS]
    assert max(values) - min(values) >= 30, f"{field} barely moves: {values}"


def test_scores_use_most_of_the_available_range(results):
    scores = list(_scores(results).values())
    assert max(scores) - min(scores) >= 400


def test_no_persona_is_pinned_to_a_score_bound(results):
    """Clamping at 900 makes two different profiles indistinguishable and
    means the model has no headroom left above its best example."""
    for persona, (_, _, _, report) in results.items():
        assert report.score < cfg.SCORE_MAX, f"{persona} is clamped at the ceiling"
        assert report.score > cfg.SCORE_MIN, f"{persona} is clamped at the floor"


def test_every_band_is_reachable(results):
    """A band no persona can reach is an untested region of the scale."""
    bands = {results[p][3].band for p in PERSONAS}
    assert bands == {"Poor", "Fair", "Good", "Very Good", "Excellent"}


def test_cashflow_block_can_penalise_as_well_as_reward(results):
    """It used to contribute +105..+199 to every persona — a flat bias dressed
    up as a ±300 block."""
    totals = {p: _cashflow_block(results[p][0])[0] for p in PERSONAS}
    assert min(totals.values()) < -50, f"cash-flow block never penalises: {totals}"
    assert max(totals.values()) > 50


def test_lifestyle_block_can_penalise_as_well_as_reward(results):
    totals = {p: _lifestyle_block(results[p][1], results[p][0])[0] for p in PERSONAS}
    assert min(totals.values()) < -50, f"lifestyle block never penalises: {totals}"
    assert max(totals.values()) > 50


def test_bounce_penalty_keeps_scaling_past_two_events(results):
    """With a -80 floor, two bounces and eight bounces cost exactly the same."""
    few = results["bnpl_heavy_spender"][0]
    many = results["frequent_bouncer"][0]
    assert many.bounce_count > few.bounce_count
    few_pts = _cashflow_block(few)[0]
    many_pts = _cashflow_block(many)[0]
    assert many_pts < few_pts


def test_high_discretionary_spend_is_not_rewarded(results):
    """L2 measures how much goes to discretionary categories. Scored on the
    same "higher is better" ramp as the other indices, it handed points to the
    persona spending 68% of income on food delivery and shopping."""
    aspirational = results["aspirational_overspender"]
    ordinary = results["balanced_salaried"]
    assert aspirational[1].l2_aspirational > ordinary[1].l2_aspirational
    assert aspirational[3].score < ordinary[3].score


# ---------------------------------------------------------------------------
# Report shape sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("persona", list(PERSONAS))
def test_report_is_well_formed(results, persona):
    _, profile, archetype, report = results[persona]
    assert cfg.SCORE_MIN <= report.score <= cfg.SCORE_MAX
    assert report.band in {"Poor", "Fair", "Good", "Very Good", "Excellent"}
    assert report.archetype == archetype
    assert report.lifestyle.archetype == archetype  # nested field kept in sync
    assert 1 <= len(report.factors) <= 6
    assert report.narrative  # never blank


@pytest.mark.parametrize("persona", list(PERSONAS))
def test_every_archetype_has_its_own_narrative(results, persona):
    """A missing entry silently falls back to the "Balanced" copy, which would
    describe an Overextended profile as unremarkable."""
    _, _, archetype, report = results[persona]
    if archetype != "Balanced":
        assert report.narrative != "No single spending pattern dominates this statement."
