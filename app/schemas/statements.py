"""Pydantic contracts for CreditMitra.

ISSUE #1 — FROZEN after merge. Every other track codes against these.
Additive only: existing Transaction fields are untouched so extraction.py and
the dashboard keep working.
"""

from __future__ import annotations

# NOTE: imported as _date, NOT `date`.
# Transaction has a field literally named `date` (the raw string from Docling).
# A plain `from datetime import date` gets shadowed by that field inside the class
# namespace, and because of `from __future__ import annotations` every annotation is
# evaluated lazily as a string — so `txn_date: date | None` resolves `date` to the
# field default "" and blows up with:
#     TypeError: unsupported operand type(s) for |: 'str' and 'NoneType'
from datetime import date as _date
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Taxonomy value types (see docs/taxonomy.md — Issue #4)
# ---------------------------------------------------------------------------

RiskFlag = Literal["gambling", "bnpl_lending", "crypto"]
LifestyleDim = Literal[
    "essential",  # feeds L1
    "aspirational",  # feeds L2
    "commitment",  # feeds L4
    "leverage",  # feeds L5
    "risk",  # feeds L6
    "neutral",  # feeds none
]
RecurringType = Literal[
    "adhoc",  # one-off
    "subscription",  # OTT, gym, telecom plan
    "emi_like",  # loan EMI, rent, insurance premium
    "payout_source",  # gig platform paying the USER (a credit) -> Gig Hustler
]
Direction = Literal["credit", "debit"]
ScoreBlock = Literal["cashflow", "lifestyle"]


# ---------------------------------------------------------------------------
# Transaction  (EXTENDED — original fields unchanged)
# ---------------------------------------------------------------------------


class Transaction(BaseModel):
    """One statement row.

    Original string fields come straight from Docling. The parsed/enrichment
    fields are populated later in the pipeline and are all optional so a
    partially-processed transaction is still a valid Transaction.
    """

    # --- original (Docling output) — DO NOT RENAME, the dashboard reads these
    date: str = ""
    particulars: str = ""
    deposits: str = ""
    withdrawals: str = ""
    balance: str = ""

    # --- payee extraction (DP SLM) + classification (ONNX) ---
    payee: str = ""
    payee_type: str | None = None  # "person" | "merchant"
    payee_confidence: float | None = None

    # --- parsed numerics (Issue #5) ---
    txn_date: _date | None = None
    amount: float | None = None
    direction: Direction | None = None
    balance_val: float | None = None

    # --- merchant enrichment (Issue #7) ---
    category: str | None = None
    is_essential: bool | None = None
    risk_flag: RiskFlag | None = None
    lifestyle_dim: LifestyleDim | None = None
    recurring_type: RecurringType | None = None


# ---------------------------------------------------------------------------
# Merchant enrichment  (Issue #7 / #7b — also the merchant-server contract)
# ---------------------------------------------------------------------------


class MerchantEnrichment(BaseModel):
    """What a merchant lookup returns. The ONLY thing that crosses the
    device boundary is the merchant NAME (see docs/enrichment_api.md)."""

    canonical_name: str = Field(description="Clean merchant name, e.g. 'Swiggy'")
    category: str = Field(description="One value from docs/taxonomy.md")
    is_essential: bool = Field(description="Counts toward L1 Essential Stability")
    risk_flag: RiskFlag | None = Field(
        default=None, description="gambling | bnpl_lending | crypto | null"
    )
    lifestyle_dim: LifestyleDim = Field(description="Which L-index this feeds")
    recurring_type: RecurringType = Field(description="Recurrence pattern")

    @classmethod
    def unknown(cls, name: str) -> MerchantEnrichment:
        """Safe default. Every failure path returns this — never raise,
        never block the pipeline."""
        return cls(
            canonical_name=name,
            category="other",
            is_essential=False,
            risk_flag=None,
            lifestyle_dim="neutral",
            recurring_type="adhoc",
        )


# ---------------------------------------------------------------------------
# Risk brain  (Track B — Issues #10-#13)
# ---------------------------------------------------------------------------


class FeatureVector(BaseModel):
    """Cash-flow features. Output of build_features(transactions)."""

    salary_detected: bool = False
    monthly_income: float = 0.0
    foir: float = 0.0  # fixed obligations / income
    net_cashflow: float = 0.0
    balance_buffer_days: float = 0.0

    essential_ratio: float = 0.0
    discretionary_ratio: float = 0.0
    cash_withdrawal_ratio: float = 0.0
    merchant_resolution_rate: float = 0.0  # share of debits with a resolved merchant

    bounce_count: int = 0
    bnpl_merchant_count: int = 0
    bnpl_share: float = 0.0
    gambling_share: float = 0.0

    months_covered: int = 0
    txn_count: int = 0


class LifestyleProfile(BaseModel):
    """The core IP. All indices 0-100. See RESEARCH_LIFESTYLE_CREDIT.md §2."""

    l1_essential_stability: int = 0
    l2_aspirational: int = 0
    l3_digital_maturity: int = 0
    l4_commitment: int = 0  # the self-control proxy — most important
    l5_leverage: int = 0  # inverse: high = LOW leverage
    l6_risk_appetite: int = 0  # inverse: high = LOW risk appetite

    # behavioural texture (Gladstone et al., EPJ Data Science 2021)
    category_diversity: float = 0.0  # normalized entropy
    merchant_loyalty: float = 0.0  # repeat-merchant txn share
    burstiness: float = 0.0  # Goh-Barabasi B = (sigma-mu)/(sigma+mu)
    category_turnover: float = 0.0  # new categories per month

    archetype: str = "unknown"


class ScoreFactor(BaseModel):
    """One human-readable reason for points gained/lost."""

    factor: str  # "Commitment Index 78 — sustained SIP + insurance + rent"
    impact: int  # signed points, e.g. +45
    block: ScoreBlock


class CreditRiskReport(BaseModel):
    """The deliverable. Rendered by LifestyleReport.tsx (Issue #17)."""

    score: int = Field(ge=300, le=900)
    band: str  # Poor | Fair | Good | Very Good | Excellent
    archetype: str
    factors: list[ScoreFactor] = Field(default_factory=list)
    lifestyle: LifestyleProfile
    features: FeatureVector
    narrative: str = ""  # one-line summary for the archetype card


# ---------------------------------------------------------------------------
# API responses  (names must match app/api/v1/statements.py imports)
# ---------------------------------------------------------------------------


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """Extraction-page shape: task status + transactions with payee/merchant
    classification. Deliberately does NOT carry the lifestyle report — see
    ReportResponse. Splitting these means the extraction page never waits on
    (or pays the payload size of) scoring data it doesn't show."""

    task_id: str
    status: str
    transactions: list[Transaction] | None = None
    error: str | None = None


class ReportResponse(BaseModel):
    """Report-page shape: GET /api/v1/statements/report/{task_id}.

    `report` is None whenever there isn't one yet — either the task hasn't
    reached "completed" (still processing) or scoring failed silently
    (Issue #14: scoring is a nice-to-have, never a completion requirement).
    `status` tells the caller which case it is; a missing/unknown task_id is
    a 404 at the route level, not a null report here.
    """

    task_id: str
    status: str
    report: CreditRiskReport | None = None
    error: str | None = None


# Backwards-compatible alias in case anything imports the other name.
ProcessResponse = TaskResponse
