"""CLI for the incremental-lift experiment.

python -m experiments.incremental_lift.run synthetic --mode signal
python -m experiments.incremental_lift.run synthetic --mode null
python -m experiments.incremental_lift.run berka --data-dir path/to/pkdd99
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import Dataset
from .featurize import ARM_CASHFLOW, ARM_MERCHANT, ARM_PAYEE, ARMS, build_matrix
from .modeling import Comparison, compare, cross_val_predictions

_PAIRS = (
    (ARM_CASHFLOW, ARM_MERCHANT, "whole merchant stack"),
    (ARM_CASHFLOW, ARM_PAYEE, "payee extraction + classification only"),
    (ARM_PAYEE, ARM_MERCHANT, "enrichment tier on top"),
)


def run(dataset: Dataset, n_folds: int = 5, l2: float | None = None) -> dict[str, object]:
    print(dataset.summary())
    for caveat in dataset.caveats:
        print(f"  caveat: {caveat}")
    print()

    if dataset.n_defaults < 2 * n_folds:
        print(
            f"WARNING: only {dataset.n_defaults} defaults for {n_folds} folds — "
            "estimates will be unstable.",
            file=sys.stderr,
        )

    matrices = {arm: build_matrix(dataset.statements, arm) for arm in ARMS}
    predictions = {
        arm: cross_val_predictions(m.x, m.y, n_folds=n_folds, l2=l2) for arm, m in matrices.items()
    }

    print(f"{'arm':<24}{'features':>9}{'AUC':>8}{'KS':>8}")
    print("-" * 49)
    from .modeling import auc, ks_statistic

    for arm in ARMS:
        p = predictions[arm]
        print(
            f"{arm:<24}{len(matrices[arm].names):>9}"
            f"{auc(p.y_true, p.y_score):>8.3f}{ks_statistic(p.y_true, p.y_score):>8.3f}"
        )
    print()

    results: dict[str, object] = {
        "dataset": dataset.name,
        "n": dataset.n,
        "n_defaults": dataset.n_defaults,
        "caveats": list(dataset.caveats),
        "arms": {arm: matrices[arm].names for arm in ARMS},
        "comparisons": {},
    }

    print(f"{'comparison':<40}{'dAUC':>8}{'95% CI':>18}{'p':>8}")
    print("-" * 74)
    comparisons: dict[str, dict[str, float | bool]] = {}
    for base_arm, treat_arm, label in _PAIRS:
        c: Comparison = compare(predictions[base_arm], predictions[treat_arm])
        key = f"{base_arm}->{treat_arm}"
        comparisons[key] = {
            "label": label,
            "baseline_auc": round(c.baseline_auc, 4),
            "treatment_auc": round(c.treatment_auc, 4),
            "delta_auc": round(c.delta_auc, 4),
            "ci_low": round(c.delta_ci_low, 4),
            "ci_high": round(c.delta_ci_high, 4),
            "delta_ks": round(c.delta_ks, 4),
            "p_value": round(c.p_value, 4),
            "significant": c.significant,
        }
        marker = "*" if c.significant else " "
        print(
            f"{label:<40}{c.delta_auc:>+8.3f}"
            f"  [{c.delta_ci_low:>+6.3f}, {c.delta_ci_high:>+6.3f}]{c.p_value:>8.3f}{marker}"
        )

    results["comparisons"] = comparisons
    print("\n* 95% CI excludes zero")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)

    syn = sub.add_parser("synthetic", help="controlled dataset — validates the harness")
    syn.add_argument("--mode", choices=["signal", "null"], default="signal")
    syn.add_argument("--n", type=int, default=600)

    ber = sub.add_parser("berka", help="PKDD'99 financial dataset — the real measurement")
    ber.add_argument("--data-dir", required=True, type=Path)
    ber.add_argument(
        "--no-sanction-distress",
        action="store_true",
        help="withhold sanction interest from the cash-flow baseline (see berka.py)",
    )

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--l2",
        type=float,
        default=None,
        help="fix the L2 penalty instead of tuning it per arm by nested CV",
    )
    parser.add_argument("--json-out", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.source == "synthetic":
        from .synthetic import make_dataset

        dataset = make_dataset(n=args.n, mode=args.mode)
    else:
        from . import berka

        dataset = berka.load(
            args.data_dir, treat_sanction_as_distress=not args.no_sanction_distress
        )

    results = run(dataset, n_folds=args.folds, l2=args.l2)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
