import contextlib
import logging
import os
import re
from typing import Any

import httpx
from docling.document_converter import DocumentConverter

from app.core.config import settings
from app.schemas.statements import Transaction
from app.services.archetype import classify_archetype
from app.services.credit_scorer import score
from app.services.feature_engineering import build_features
from app.services.lifestyle_profile import build_profile
from app.services.merchant_enrichment import get_enrichment_service
from app.services.parsing import (
    derive_direction,
    is_junk_row,
    parse_amount,
    parse_date,
    strip_chq_artifacts,
)
from app.services.task_store import update_task_status

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are an information extraction model. Extract only the payee name "
    "from the transaction narration. Return only the payee text, with no extra words."
)


def build_prompt(narration: str) -> str:
    return f"{SYSTEM_INSTRUCTION}\n\nTransaction narration:\n{narration}\n\nPayee:"


def predict_payee(narration: str, client: httpx.Client) -> str:
    """Stateless payee extraction via Ollama (no conversation context)."""
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": build_prompt(narration),
        "stream": False,
        "raw": True,
        "options": {"temperature": 0, "num_predict": 32},
    }
    try:
        resp = client.post(f"{settings.OLLAMA_HOST}/api/generate", json=payload, timeout=60.0)
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()
    except Exception as e:
        logger.error(f"Ollama prediction failed: {e}")
        return ""


def normalize_narration(text: str) -> str:
    """Collapse multi-line narrations into one line; segments join with no gap."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [line.strip() for line in text.split("\n") if line.strip()]
    if parts:
        return "".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def extract_transactions(pdf_path: str) -> list[dict[str, Any]]:
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    rows: list[dict[str, Any]] = []
    for table in result.document.tables:
        df = table.export_to_dataframe()
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.fillna("")

        for row in df.to_dict("records"):
            particulars_raw = normalize_narration(str(row.get("particulars", "")))
            rec: dict[str, Any] = {
                "date": str(row.get("date", "")).strip(),
                "particulars": particulars_raw,
                "deposits": str(row.get("deposits", "")).strip(),
                "withdrawals": str(row.get("withdrawals", "")).strip(),
                "balance": str(row.get("balance", "")).strip(),
            }
            # is_junk_row must see the ORIGINAL particulars (incl. any "Chq: <ref>"
            # bleed) — that's exactly the pattern it's matching on. Strip after.
            if not any(rec.values()) or is_junk_row(rec):
                continue
            rec["particulars"] = strip_chq_artifacts(particulars_raw)

            txn_date = parse_date(rec["date"])
            rec["txn_date"] = txn_date.isoformat() if txn_date else None

            direction_amount = derive_direction(rec["deposits"], rec["withdrawals"])
            rec["direction"], rec["amount"] = direction_amount if direction_amount else (None, None)

            rec["balance_val"] = parse_amount(rec["balance"])

            rows.append(rec)
    return rows


def process_pdf_task(task_id: str, pdf_path: str) -> None:
    """Background task to process the PDF and update the task store."""
    try:
        update_task_status(task_id, "processing")

        # 1. Extract tables via Docling
        transactions = extract_transactions(pdf_path)

        # 2. Enrich via Ollama
        with httpx.Client(timeout=120.0) as client:
            for txn in transactions:
                narration = txn.get("particulars", "")
                if narration and narration not in ("Opening Balance", "Closing Balance"):
                    txn["payee"] = predict_payee(narration, client)
                else:
                    txn["payee"] = ""

        # 3. Classify payees as person/merchant
        from app.services.merchant_classifier import MerchantClassifierService

        classifier = MerchantClassifierService.get_instance()
        if classifier.available:
            payee_names = [txn.get("payee", "") for txn in transactions]
            classifications = classifier.classify_batch(payee_names)
            for txn, cls_result in zip(transactions, classifications, strict=False):
                txn["payee_type"] = cls_result["label"]
                txn["payee_confidence"] = cls_result["confidence"]
        else:
            logger.warning("Merchant classifier not available — skipping classification")
            for txn in transactions:
                txn["payee_type"] = None
                txn["payee_confidence"] = None

        # 4. Enrich merchant payees (Issue #7 / #7b) — cache -> dictionary -> LLM.
        # Only merchant-classified payees are sent; enrich() never raises.
        merchant_names = [
            txn.get("payee", "") for txn in transactions if txn.get("payee_type") == "merchant"
        ]
        if merchant_names:
            enrichments = iter(get_enrichment_service().enrich(merchant_names))
            for txn in transactions:
                if txn.get("payee_type") == "merchant":
                    enrichment = next(enrichments)
                    txn["category"] = enrichment.category
                    txn["is_essential"] = enrichment.is_essential
                    txn["risk_flag"] = enrichment.risk_flag
                    txn["lifestyle_dim"] = enrichment.lifestyle_dim
                    txn["recurring_type"] = enrichment.recurring_type

        # 5. Score (Track B, Issues #10-13): build_features -> build_profile ->
        # classify_archetype -> score. Never allowed to fail the whole task —
        # a report is a nice-to-have on top of the transactions, not a
        # precondition for "completed".
        report_payload = None
        try:
            txn_models = [Transaction(**txn) for txn in transactions]
            features = build_features(txn_models)
            profile = build_profile(txn_models, features)
            archetype = classify_archetype(features, profile)
            report_payload = score(features, profile, archetype).model_dump()
        except Exception as exc:
            logger.warning("Scoring failed for task %s: %s", task_id, exc)

        # 6. Mark completed and save results
        update_task_status(task_id, "completed", transactions=transactions, report=report_payload)

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        update_task_status(task_id, "failed", error=str(e))
    finally:
        # Cleanup temp file
        if os.path.exists(pdf_path):
            with contextlib.suppress(Exception):
                os.unlink(pdf_path)
