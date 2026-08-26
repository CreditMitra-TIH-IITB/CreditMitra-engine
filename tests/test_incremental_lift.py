"""The incremental-lift harness, and the two controls that validate it.

A measurement instrument that has not been shown to (a) find real signal and
(b) refuse to invent fake signal is not evidence of anything. Both controls run
here so that a change to featurisation, masking or the evaluator has to keep
clearing them.

Sample sizes are smaller than the numbers quoted in the README so the suite
stays fast; the effects are large enough to survive the reduction.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.incremental_lift.featurize import (
    ARM_CASHFLOW,
    ARM_MERCHANT,
    ARM_PAYEE,
    build_matrix,
    mask_for_arm,
)
from experiments.incremental_lift.modeling import (
    auc,
    compare,
    cross_val_predictions,
    ks_statistic,
    stratified_folds,
)
from experiments.incremental_lift.synthetic import make_dataset

FOLDS = 5
BOOTSTRAP = 400
N = 400


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_auc_is_one_for_perfect_separation():
    y = np.array([0, 0, 1, 1], dtype=float)
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_auc_is_half_for_constant_scores():
    """Ties must average to 0.5, not silently resolve to 1.0 by sort order —
    a constant-score model is the definition of no information."""
    y = np.array([0, 1, 0, 1], dtype=float)
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_auc_is_symmetric_under_score_inversion():
    y = np.array([0, 0, 1, 1], dtype=float)
    s = np.array([0.1, 0.4, 0.6, 0.9])
    assert auc(y, s) + auc(y, -s) == pytest.approx(1.0)


def test_auc_is_defined_when_a_fold_is_single_class():
    y = np.array([1, 1, 1], dtype=float)
    assert auc(y, np.array([0.1, 0.5, 0.9])) == 0.5


def test_ks_is_zero_for_constant_scores():
    y = np.array([0, 1, 0, 1], dtype=float)
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_stratified_folds_all_contain_defaults():
    """At a ~15% base rate an unstratified split routinely yields a fold with no
    defaults, where AUC is undefined."""
    y = np.array([1] * 30 + [0] * 170, dtype=float)
    for fold in stratified_folds(y, FOLDS):
        assert y[fold].sum() > 0


# ---------------------------------------------------------------------------
# Masking — the property the whole ablation rests on
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def signal_dataset():
    return make_dataset(n=N, mode="signal")


@pytest.fixture(scope="module")
def null_dataset():
    return make_dataset(n=N, mode="null", seed=7761)


def test_cashflow_arm_cannot_see_payee_or_enrichment(signal_dataset):
    txns = signal_dataset.statements[0].transactions
    masked = mask_for_arm(txns, ARM_CASHFLOW)
    assert any(t.payee for t in txns), "fixture should have payees to hide"
    for t in masked:
        assert t.payee == ""
        assert t.payee_type is None
        assert t.category is None
        assert t.lifestyle_dim is None
        assert t.risk_flag is None


def test_payee_arm_keeps_identity_but_not_enrichment(signal_dataset):
    txns = signal_dataset.statements[0].transactions
    masked = mask_for_arm(txns, ARM_PAYEE)
    assert any(t.payee for t in masked)
    assert all(t.category is None for t in masked)


def test_masking_does_not_mutate_the_original(signal_dataset):
    """The arms share one list of statements; masking in place would silently
    contaminate every later arm."""
    txns = signal_dataset.statements[0].transactions
    before = [(t.payee, t.category) for t in txns]
    mask_for_arm(txns, ARM_CASHFLOW)
    assert [(t.payee, t.category) for t in txns] == before


def test_arms_are_nested_in_information(signal_dataset):
    counts = {
        arm: len(build_matrix(signal_dataset.statements, arm).names)
        for arm in (ARM_CASHFLOW, ARM_PAYEE, ARM_MERCHANT)
    }
    assert counts[ARM_CASHFLOW] < counts[ARM_PAYEE] < counts[ARM_MERCHANT]


def test_features_are_finite(signal_dataset):
    matrix = build_matrix(signal_dataset.statements, ARM_MERCHANT)
    assert np.isfinite(matrix.x).all()
    assert matrix.x.shape[0] == signal_dataset.n


# ---------------------------------------------------------------------------
# The controls
# ---------------------------------------------------------------------------


def _run_pair(dataset, base_arm, treat_arm):
    base = build_matrix(dataset.statements, base_arm)
    treat = build_matrix(dataset.statements, treat_arm)
    base_pred = cross_val_predictions(base.x, base.y, n_folds=FOLDS)
    treat_pred = cross_val_predictions(treat.x, treat.y, n_folds=FOLDS)
    return compare(base_pred, treat_pred, n_bootstrap=BOOTSTRAP)


# Nested CV is the expensive part of the suite, so each comparison is computed
# once per module rather than once per assertion.
@pytest.fixture(scope="module")
def signal_comparison(signal_dataset):
    return _run_pair(signal_dataset, ARM_CASHFLOW, ARM_MERCHANT)


@pytest.fixture(scope="module")
def null_comparison(null_dataset):
    return _run_pair(null_dataset, ARM_CASHFLOW, ARM_MERCHANT)


@pytest.fixture(scope="module")
def blind_arm_auc(signal_dataset):
    matrix = build_matrix(signal_dataset.statements, ARM_CASHFLOW)
    predictions = cross_val_predictions(matrix.x, matrix.y, n_folds=FOLDS)
    return auc(predictions.y_true, predictions.y_score)


def test_positive_control_recovers_planted_merchant_signal(signal_comparison):
    """Every borrower spends the same rupees on the same dates; only the
    merchant behind each payment differs, and the outcome depends on it. An arm
    that can read merchants must beat one that cannot."""
    result = signal_comparison
    assert result.delta_auc > 0.10
    assert result.delta_ci_low > 0, "planted signal should be unambiguous"


def test_positive_control_leaves_the_blind_arm_near_chance(blind_arm_auc):
    """If Arm A scores well here, the signal has leaked through the cash-flow
    footprint and the control proves nothing. An earlier generator scaled risky
    spend on top of normal spend and Arm A reached 0.856."""
    assert blind_arm_auc < 0.62


def test_null_control_fabricates_no_lift(null_comparison):
    """Arm C carries ~15 more columns of pure noise. Any leak in the evaluator —
    a shared standardiser, in-fold scoring, an unpenalised fit — turns that into
    spurious lift, which would void every real-data number this harness reports.

    The assertion is one-sided on purpose. A small *negative* delta is the
    correct result, not a failure: extra parameters cost estimation variance,
    and at n=400 with ~76 defaults the penalty measures about -0.03 AUC even
    with the penalty tuned per arm. What must never happen is noise reading as
    improvement.
    """
    result = null_comparison
    assert result.delta_ci_low <= 0, (
        f"fabricated lift on noise: {result.delta_auc:+.3f} "
        f"[{result.delta_ci_low:+.3f}, {result.delta_ci_high:+.3f}]"
    )


def test_small_sample_penalty_is_bounded(null_comparison):
    """Guards the caveat that matters for Berka. The dimensionality penalty on
    pure noise has to stay small enough that a genuine effect could still clear
    it — if noise alone cost 0.15 AUC, a null result on real data would be
    uninterpretable rather than informative."""
    assert null_comparison.delta_auc > -0.10


def test_null_control_baseline_still_discriminates(null_dataset):
    """Sanity: in null mode cash-flow stress genuinely drives the outcome, so a
    flat 0.5 for Arm A would mean the features are broken rather than that the
    control passed."""
    matrix = build_matrix(null_dataset.statements, ARM_CASHFLOW)
    predictions = cross_val_predictions(matrix.x, matrix.y, n_folds=FOLDS)
    assert auc(predictions.y_true, predictions.y_score) > 0.60


def test_comparison_reports_sample_counts(signal_dataset, signal_comparison):
    result = signal_comparison
    assert result.n == signal_dataset.n
    assert result.n_defaults == signal_dataset.n_defaults
    assert 0 < result.n_defaults < result.n
