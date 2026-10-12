"""Guards found by running real statements end to end: person names classified
as merchants, invented payees, gig-less "Gig Hustlers", refunds counted as
bounces, and tasks left processing by a crash."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from app.core.config import settings
from app.schemas.statements import FeatureVector, LifestyleProfile, Transaction
from app.services import task_store
from app.services.archetype import GIG_HUSTLER, classify_archetype
from app.services.credit_scorer import _cashflow_block
from app.services.extraction import ground_payee, payee_is_grounded
from app.services.feature_engineering import build_features
from app.services.merchant_classifier import MerchantClassifierService, looks_like_person_name

# --- person-name guard --------------------------------------------------------


@pytest.mark.parametrize("name", ["Neha", "Rohit Sa", "Ramesh Kumar Sharma", "Arvind Si"])
def test_personal_names_are_recognised(name: str) -> None:
    assert looks_like_person_name(name)


@pytest.mark.parametrize(
    "name", ["Swiggy", "Ganesh Traders", "Sai Medical", "Axis Bank", "Zomato Ltd", "IRCTC"]
)
def test_merchant_names_are_not(name: str) -> None:
    assert not looks_like_person_name(name)


def test_classifier_overrides_a_confident_merchant_label_on_a_person_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MerchantClassifierService.__new__(MerchantClassifierService)
    service._available = True
    monkeypatch.setattr(
        service,
        "_onnx_classify",
        lambda name: {"label": "merchant", "confidence": 0.94, "p_merchant": 0.94},
    )

    neha, swiggy = service.classify_batch(["Neha", "Swiggy"])

    assert neha["label"] == "person"
    assert swiggy["label"] == "merchant"


# --- payee grounding ----------------------------------------------------------


@pytest.mark.parametrize(
    ("payee", "narration"),
    [
        ("Swiggy", "UPI/DR/601591718046/SWIGGY LI/UTIB/x"),
        ("Jio Recharge", "UPI/DR/1/JIO RECHA/X/Y"),  # expanding a truncation is fine
        ("Ramesh Kumar Sharma", "NEFT-HDFC0001234-RAMESH KUMAR SHARMA-SALARY"),
    ],
)
def test_grounded_payees_are_kept(payee: str, narration: str) -> None:
    assert payee_is_grounded(payee, narration)
    assert ground_payee(payee, narration) == payee


def test_invented_payee_falls_back_to_the_upi_name_segment() -> None:
    assert ground_payee("PayViva", "UPI/DR/601/NYKAA ERE/YESB/x@ybl") == "Nykaa Ere"
    assert ground_payee("Axie", "UPI-SUKANT SI-sukant@ok-HDFC0001-1") == "Sukant Si"


def test_invented_payee_is_kept_when_there_is_nothing_better() -> None:
    # Not a UPI narration: no reliable place to read a name from.
    assert ground_payee("PayViva", "NEFT 12345 XYZ") == "PayViva"


# --- archetype & income -------------------------------------------------------


def _profile() -> LifestyleProfile:
    return LifestyleProfile(
        l1_essential_stability=60,
        l2_aspirational=30,
        l3_digital_maturity=90,
        l4_commitment=50,
        l5_leverage=100,
        l6_risk_appetite=100,
    )


def test_inflows_without_gig_payouts_are_not_a_gig_hustler() -> None:
    family_support = FeatureVector(salary_detected=False, monthly_income=8000, months_covered=3)
    assert classify_archetype(family_support, _profile()) != GIG_HUSTLER

    _, factors = _cashflow_block(family_support)
    assert factors[0].impact == 0


def test_gig_payouts_still_make_a_gig_hustler() -> None:
    gig = FeatureVector(
        salary_detected=False, gig_income_detected=True, monthly_income=30000, months_covered=3
    )
    assert classify_archetype(gig, _profile()) == GIG_HUSTLER


# --- bounces vs refunds -------------------------------------------------------


def _txn(particulars: str, direction: str, day: int) -> Transaction:
    return Transaction(
        particulars=particulars,
        txn_date=date(2026, 1, day),
        amount=500.0,
        direction=direction,  # type: ignore[arg-type]
        balance_val=10000.0,
    )


def test_refunds_are_not_bounces() -> None:
    rows = [
        _txn("UPI/CR/123/AMAZON/RETURN REFUND", "credit", 2),
        _txn("IMPS RETURN TO SENDER REFUND", "credit", 3),
        _txn("UPI REVERSAL 98765", "credit", 4),
        _txn("ECS RET CHRG/BAJAJ FINANCE EMI/INSUFFICIENT FUNDS", "debit", 5),
    ]
    assert build_features(rows).bounce_count == 1


# --- crash recovery -----------------------------------------------------------


def test_interrupted_tasks_are_failed_and_their_uploads_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    (tmp_path / "tasks").mkdir()
    (tmp_path / "uploads").mkdir()

    def make(task_id: str, status: str) -> Path:
        (tmp_path / "tasks" / f"{task_id}.json").write_text(
            json.dumps({"task_id": task_id, "status": status})
        )
        upload = tmp_path / "uploads" / f"{task_id}.pdf"
        upload.write_bytes(b"%PDF")
        return upload

    stuck_upload = make("stuck", "processing")
    done_upload = make("done", "completed")

    assert task_store.fail_interrupted_tasks() == 1

    stuck: dict[str, Any] = json.loads((tmp_path / "tasks" / "stuck.json").read_text())
    assert stuck["status"] == "failed"
    assert stuck["error"] == task_store.INTERRUPTED_ERROR
    assert not stuck_upload.exists()
    assert done_upload.exists()  # finished tasks' files are never touched
