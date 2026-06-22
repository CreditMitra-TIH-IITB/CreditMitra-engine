import contextlib
import logging
import os
import re
from typing import Any

import httpx
from docling.document_converter import DocumentConverter

from app.core.config import settings
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

        for _, row in df.iterrows():
            rec = {
                "date": str(row.get("date", "")).strip(),
                "particulars": normalize_narration(str(row.get("particulars", ""))),
                "deposits": str(row.get("deposits", "")).strip(),
                "withdrawals": str(row.get("withdrawals", "")).strip(),
                "balance": str(row.get("balance", "")).strip(),
            }
            if not any(rec.values()):
                continue
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
            for txn, cls_result in zip(transactions, classifications):
                txn["payee_type"] = cls_result["label"]
                txn["payee_confidence"] = cls_result["confidence"]
        else:
            logger.warning("Merchant classifier not available — skipping classification")
            for txn in transactions:
                txn["payee_type"] = None
                txn["payee_confidence"] = None

        # 4. Mark completed and save results
        update_task_status(task_id, "completed", transactions=transactions)

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        update_task_status(task_id, "failed", error=str(e))
    finally:
        # Cleanup temp file
        if os.path.exists(pdf_path):
            with contextlib.suppress(Exception):
                os.unlink(pdf_path)
