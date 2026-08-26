"""Scoring model, discrimination metrics, and the paired comparison.

Deliberately numpy-only. Two reasons:

1. No new dependency on the engine for what is a research side-experiment.
2. An additive logistic model *is* the regulated-industry standard for
   scorecards (WoE/logistic), so this is not a simplification we are apologising
   for — it is the right model class, and keeping it in ~80 readable lines means
   a reviewer can audit exactly what produced the reported number.

The comparison that matters is paired: both feature sets see identical folds and
identical borrowers, so the difference in AUC is attributable to the features and
not to a lucky split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RNG_SEED = 20260826


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUC, ties averaged.

    Returns 0.5 when a fold contains only one class — undefined rather than
    informative, and 0.5 is the honest "no information" value.
    """
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # Average ranks within tied score groups, or ties inflate the statistic.
    sorted_scores = y_score[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation — the number Indian credit teams quote
    alongside AUC."""
    positives = y_true == 1
    if positives.sum() == 0 or (~positives).sum() == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted, s_sorted = y_true[order], y_score[order]
    cum_pos = np.cumsum(y_sorted == 1) / max(int((y_sorted == 1).sum()), 1)
    cum_neg = np.cumsum(y_sorted == 0) / max(int((y_sorted == 0).sum()), 1)
    separation = np.abs(cum_pos - cum_neg)

    # Only compare the two curves at genuine score thresholds. Inside a run of
    # tied scores the sort order is arbitrary, and reading the gap mid-run
    # reports separation the model cannot actually make — a constant-score model
    # would score a healthy KS purely on tie-break order.
    at_threshold = np.ones(len(s_sorted), dtype=bool)
    at_threshold[:-1] = s_sorted[:-1] != s_sorted[1:]
    return float(separation[at_threshold].max())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class LogisticScorecard:
    """L2-regularised logistic regression, full-batch gradient descent.

    `l2` is not optional. With a few hundred borrowers and tens of correlated
    features, the unregularised fit separates the training folds perfectly and
    reports a meaningless AUC — which is exactly the failure mode this whole
    experiment exists to avoid.
    """

    l2: float = 1.0
    iterations: int = 2000
    learning_rate: float = 0.1

    def __post_init__(self) -> None:
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def _standardize(self, x: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self._mean = x.mean(axis=0)
            scale = x.std(axis=0)
            # A constant column carries no information; leave it at zero rather
            # than dividing by ~0 and manufacturing a huge spurious feature.
            scale[scale < 1e-9] = 1.0
            self._scale = scale
        assert self._mean is not None and self._scale is not None
        return (x - self._mean) / self._scale

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticScorecard:
        xs = self._standardize(np.asarray(x, dtype=float), fit=True)
        y = np.asarray(y, dtype=float)
        n, d = xs.shape

        w = np.zeros(d)
        b = 0.0
        for _ in range(self.iterations):
            z = xs @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            residual = p - y
            grad_w = xs.T @ residual / n + self.l2 * w / n
            grad_b = float(residual.mean())
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        self._weights, self._bias = w, b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("fit() first")
        xs = self._standardize(np.asarray(x, dtype=float), fit=False)
        z = xs @ self._weights + self._bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------------------
# Cross-validated, paired evaluation
# ---------------------------------------------------------------------------


def stratified_folds(y: np.ndarray, n_folds: int, seed: int = RNG_SEED) -> list[np.ndarray]:
    """Stratified so every fold carries defaults. With base rates around 10%,
    an unstratified split routinely produces a fold with none, and AUC there is
    undefined."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for position, sample in enumerate(idx):
            folds[position % n_folds].append(int(sample))
    return [np.array(sorted(f)) for f in folds]


@dataclass(frozen=True)
class FoldPredictions:
    y_true: np.ndarray
    y_score: np.ndarray


#: Searched per arm inside each training fold — see cross_val_predictions.
L2_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def _select_l2(
    x: np.ndarray, y: np.ndarray, grid: tuple[float, ...], n_folds: int, seed: int
) -> float:
    """Pick the penalty by inner cross-validation on the training fold only."""
    if len(grid) == 1:
        return grid[0]
    inner = stratified_folds(y, n_folds, seed)
    best_l2, best_auc = grid[0], -1.0
    for candidate in grid:
        scores = np.zeros(len(y), dtype=float)
        for held_out in inner:
            mask = np.ones(len(y), dtype=bool)
            mask[held_out] = False
            if len(np.unique(y[mask])) < 2:
                continue
            model = LogisticScorecard(l2=candidate).fit(x[mask], y[mask])
            scores[held_out] = model.predict_proba(x[held_out])
        candidate_auc = auc(y, scores)
        if candidate_auc > best_auc:
            best_l2, best_auc = candidate, candidate_auc
    return best_l2


def cross_val_predictions(
    x: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    l2: float | None = None,
    seed: int = RNG_SEED,
    inner_folds: int = 3,
) -> FoldPredictions:
    """Out-of-fold predictions, with the penalty tuned by *nested* CV.

    Fixing one penalty across arms looks fair and is not. A wider arm needs more
    shrinkage, and denying it that charges the extra columns a variance penalty
    that has nothing to do with whether they carry signal. On the null control
    at n=400 a fixed penalty made Arm C look 0.04 AUC *worse* than Arm A purely
    for being wider — and Berka sits at exactly that sample size, so the
    real-data comparison would have been biased against finding lift.

    Tuning happens inside the training fold only, so the held-out borrowers stay
    unseen and the reported AUC keeps its meaning.
    """
    grid = L2_GRID if l2 is None else (l2,)
    folds = stratified_folds(y, n_folds, seed)
    scores = np.zeros(len(y), dtype=float)
    for i, held_out in enumerate(folds):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[held_out] = False
        x_train, y_train = x[train_mask], y[train_mask]
        chosen = _select_l2(x_train, y_train, grid, inner_folds, seed + i + 1)
        model = LogisticScorecard(l2=chosen).fit(x_train, y_train)
        scores[held_out] = model.predict_proba(x[held_out])
    return FoldPredictions(y_true=y, y_score=scores)


@dataclass(frozen=True)
class Comparison:
    baseline_auc: float
    treatment_auc: float
    delta_auc: float
    delta_ci_low: float
    delta_ci_high: float
    baseline_ks: float
    treatment_ks: float
    delta_ks: float
    p_value: float
    n: int
    n_defaults: int

    @property
    def significant(self) -> bool:
        """CI excluding zero. Reported rather than asserted — a null result here
        is a finding, not a failure."""
        return self.delta_ci_low > 0 or self.delta_ci_high < 0


def compare(
    baseline: FoldPredictions,
    treatment: FoldPredictions,
    n_bootstrap: int = 2000,
    seed: int = RNG_SEED,
) -> Comparison:
    """Paired bootstrap over borrowers.

    Resampling *borrowers* (not predictions independently) keeps the pairing:
    each bootstrap replicate scores both feature sets on the same people, which
    is what makes the delta's confidence interval meaningful. Two separately
    computed AUC intervals would overlap wildly and say nothing about whether
    the difference is real.
    """
    y = baseline.y_true
    rng = np.random.default_rng(seed)

    base_auc = auc(y, baseline.y_score)
    treat_auc = auc(y, treatment.y_score)

    deltas = np.empty(n_bootstrap)
    n = len(y)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            deltas[i] = 0.0
            continue
        deltas[i] = auc(y[idx], treatment.y_score[idx]) - auc(y[idx], baseline.y_score[idx])

    observed = treat_auc - base_auc
    low, high = np.percentile(deltas, [2.5, 97.5])
    # Two-sided bootstrap p-value: how often the replicate delta crosses zero.
    centred = deltas - deltas.mean()
    p_value = float(2 * min((centred >= abs(observed)).mean(), (centred <= -abs(observed)).mean()))

    return Comparison(
        baseline_auc=base_auc,
        treatment_auc=treat_auc,
        delta_auc=observed,
        delta_ci_low=float(low),
        delta_ci_high=float(high),
        baseline_ks=ks_statistic(y, baseline.y_score),
        treatment_ks=ks_statistic(y, treatment.y_score),
        delta_ks=ks_statistic(y, treatment.y_score) - ks_statistic(y, baseline.y_score),
        p_value=min(p_value, 1.0),
        n=n,
        n_defaults=int(y.sum()),
    )
