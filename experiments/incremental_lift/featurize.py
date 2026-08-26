"""Three nested feature arms, built by masking inputs rather than picking columns.

The question WP4 has to answer is "what does merchant information buy us?", and
the honest way to answer it is an ablation. Rather than hand-select which
features belong to which arm — easy to get subtly wrong, and impossible for a
reviewer to verify — each arm is given a *masked copy of the transactions* and
run through the engine's real pipeline:

    Arm A  cash-flow only    payee, payee_type and all enrichment fields blanked
    Arm B  + payee identity  enrichment fields blanked
    Arm C  + merchant info   nothing blanked

Arm A cannot leak merchant knowledge because the fields are literally None. And
because every arm runs `build_features` / `build_profile` from app/services,
the numbers measured are the ones the product actually computes — not a
reimplementation that might flatter itself.

The arms are nested, so C − A is the whole merchant stack's contribution, B − A
is payee extraction and merchant/non-merchant classification alone, and C − B is
the enrichment tier on top. That decomposition is the useful result: it says
which part of the stack earns its keep.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import date

import numpy as np

from app.schemas.statements import Transaction
from app.services.feature_engineering import build_features
from app.services.lifestyle_profile import build_profile

from .contract import LabelledStatement

ARM_CASHFLOW = "A_cashflow_only"
ARM_PAYEE = "B_plus_payee_identity"
ARM_MERCHANT = "C_plus_merchant_info"
ARMS = (ARM_CASHFLOW, ARM_PAYEE, ARM_MERCHANT)

_ENRICHMENT_FIELDS = {
    "category": None,
    "is_essential": None,
    "risk_flag": None,
    "lifestyle_dim": None,
    "recurring_type": None,
}


_PAYEE_FIELDS: dict[str, object] = {"payee": "", "payee_type": None, "payee_confidence": None}


def mask_for_arm(transactions: list[Transaction], arm: str) -> list[Transaction]:
    """Blank the fields an arm is not allowed to see. Transaction is a pydantic
    model, so cloning goes through model_copy rather than dataclasses.replace."""
    if arm == ARM_MERCHANT:
        return transactions
    if arm == ARM_PAYEE:
        return [t.model_copy(update=dict(_ENRICHMENT_FIELDS)) for t in transactions]
    if arm == ARM_CASHFLOW:
        blanked = {**_ENRICHMENT_FIELDS, **_PAYEE_FIELDS}
        return [t.model_copy(update=dict(blanked)) for t in transactions]
    raise ValueError(f"unknown arm: {arm}")


# ---------------------------------------------------------------------------
# Balance and cadence statistics the FeatureVector does not carry
# ---------------------------------------------------------------------------


def _month_key(d: date) -> tuple[int, int]:
    return (d.year, d.month)


def _raw_statistics(transactions: list[Transaction]) -> dict[str, float]:
    """Pure arithmetic over amounts, dates and balances. Available to every arm
    because none of it needs to know who was paid."""
    usable = [t for t in transactions if t.txn_date and t.amount is not None]
    credits = [t.amount for t in usable if t.direction == "credit" and t.amount]
    debits = [t.amount for t in usable if t.direction == "debit" and t.amount]
    balances = [t.balance_val for t in usable if t.balance_val is not None]

    by_month: dict[tuple[int, int], float] = {}
    for t in usable:
        if t.txn_date is None or t.amount is None:
            continue
        signed = t.amount if t.direction == "credit" else -t.amount
        by_month[_month_key(t.txn_date)] = by_month.get(_month_key(t.txn_date), 0.0) + signed
    monthly_net = list(by_month.values())

    dates = sorted(t.txn_date for t in usable if t.txn_date)
    if len(dates) >= 3:
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        mu = sum(gaps) / len(gaps)
        sigma = math.sqrt(sum((g - mu) ** 2 for g in gaps) / len(gaps))
        burstiness = (sigma - mu) / (sigma + mu) if (sigma + mu) else 0.0
    else:
        burstiness = 0.0

    stats = {
        "credit_count": float(len(credits)),
        "debit_count": float(len(debits)),
        "avg_credit": float(np.mean(credits)) if credits else 0.0,
        "avg_debit": float(np.mean(debits)) if debits else 0.0,
        "credit_volatility": float(np.std(credits)) if len(credits) > 1 else 0.0,
        "min_balance": float(min(balances)) if balances else 0.0,
        "avg_balance": float(np.mean(balances)) if balances else 0.0,
        "balance_volatility": float(np.std(balances)) if len(balances) > 1 else 0.0,
        "negative_balance_rows": float(sum(1 for b in balances if b < 0)),
        "monthly_net_mean": float(np.mean(monthly_net)) if monthly_net else 0.0,
        "monthly_net_std": float(np.std(monthly_net)) if len(monthly_net) > 1 else 0.0,
        "losing_month_share": (
            float(sum(1 for v in monthly_net if v < 0) / len(monthly_net)) if monthly_net else 0.0
        ),
        "burstiness": float(burstiness),
    }
    return stats


def _payee_statistics(transactions: list[Transaction]) -> dict[str, float]:
    """Needs payee extraction and merchant/non-merchant classification, but no
    enrichment — this is what Arm B adds over Arm A."""
    debits = [t for t in transactions if t.direction == "debit"]
    named = [t for t in debits if t.payee]
    if not debits:
        return {"merchant_txn_share": 0.0, "unique_payee_ratio": 0.0, "payee_concentration": 0.0}

    merchant_rows = sum(1 for t in debits if t.payee_type == "merchant")
    names = [t.payee.strip().lower() for t in named]
    counts = Counter(names)
    top_share = (max(counts.values()) / len(names)) if names else 0.0

    return {
        "merchant_txn_share": merchant_rows / len(debits),
        "unique_payee_ratio": (len(counts) / len(names)) if names else 0.0,
        "payee_concentration": top_share,
    }


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

_FEATURE_VECTOR_FIELDS = (
    "salary_detected",
    "monthly_income",
    "foir",
    "net_cashflow",
    "balance_buffer_days",
    "essential_ratio",
    "discretionary_ratio",
    "cash_withdrawal_ratio",
    "merchant_resolution_rate",
    "bounce_count",
    "bnpl_merchant_count",
    "bnpl_share",
    "gambling_share",
    "months_covered",
    "txn_count",
)

_PROFILE_FIELDS = (
    "l1_essential_stability",
    "l2_aspirational",
    "l3_digital_maturity",
    "l4_commitment",
    "l5_leverage",
    "l6_risk_appetite",
    "category_diversity",
    "merchant_loyalty",
    "category_turnover",
)


def featurize_one(transactions: list[Transaction], arm: str) -> dict[str, float]:
    masked = mask_for_arm(transactions, arm)

    features = build_features(masked)
    row: dict[str, float] = {f: float(getattr(features, f)) for f in _FEATURE_VECTOR_FIELDS}
    row.update(_raw_statistics(masked))

    if arm in (ARM_PAYEE, ARM_MERCHANT):
        row.update(_payee_statistics(masked))

    if arm == ARM_MERCHANT:
        profile = build_profile(masked, features)
        row.update({f: float(getattr(profile, f)) for f in _PROFILE_FIELDS})

    return row


@dataclass(frozen=True)
class FeatureMatrix:
    arm: str
    names: list[str]
    x: np.ndarray
    y: np.ndarray

    def drop_constant_columns(self) -> FeatureMatrix:
        """A column that never varies cannot carry signal, and in a masked arm
        there will be several — essential_ratio is identically zero once the
        enrichment fields are blanked. Dropping them keeps the regulariser from
        spending capacity on noise and makes the reported feature count honest
        about how much information each arm really has."""
        keep = [i for i in range(self.x.shape[1]) if float(np.std(self.x[:, i])) > 1e-9]
        return FeatureMatrix(
            arm=self.arm,
            names=[self.names[i] for i in keep],
            x=self.x[:, keep],
            y=self.y,
        )


def build_matrix(statements: list[LabelledStatement], arm: str) -> FeatureMatrix:
    rows = [featurize_one(s.transactions, arm) for s in statements]
    names = sorted(rows[0])
    x = np.array([[r[name] for name in names] for r in rows], dtype=float)
    # Guard against a NaN quietly propagating into the model and out again as a
    # plausible-looking AUC.
    if not np.isfinite(x).all():
        bad = [names[i] for i in range(x.shape[1]) if not np.isfinite(x[:, i]).all()]
        raise ValueError(f"non-finite feature values in {arm}: {bad}")
    y = np.array([1 if s.defaulted else 0 for s in statements], dtype=float)
    return FeatureMatrix(arm=arm, names=names, x=x, y=y).drop_constant_columns()
