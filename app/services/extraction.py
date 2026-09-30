import logging
import re
from typing import Any

import httpx
import pdfplumber
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

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
from app.services.statement_layout import TableParseResult, rows_from_tables
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


def _process_extracted_row(
    date_val: str,
    particulars_raw: str,
    deposits_val: str,
    withdrawals_val: str,
    balance_val: str,
) -> dict[str, Any] | None:
    rec: dict[str, Any] = {
        "date": str(date_val).strip(),
        "particulars": particulars_raw,
        "deposits": str(deposits_val).strip(),
        "withdrawals": str(withdrawals_val).strip(),
        "balance": str(balance_val).strip(),
    }
    if not any(rec.values()) or is_junk_row(rec):
        return None
    rec["particulars"] = strip_chq_artifacts(particulars_raw)

    txn_date = parse_date(rec["date"])
    rec["txn_date"] = txn_date.isoformat() if txn_date else None

    direction_amount = derive_direction(rec["deposits"], rec["withdrawals"])
    rec["direction"], rec["amount"] = direction_amount if direction_amount else (None, None)

    rec["balance_val"] = parse_amount(rec["balance"])
    return rec


def _rows_to_records(parsed: TableParseResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fields in parsed.rows:
        rec = _process_extracted_row(
            fields["date"],
            normalize_narration(fields["particulars"]),
            fields["deposits"],
            fields["withdrawals"],
            fields["balance"],
        )
        if rec:
            rows.append(rec)
    return rows


def _docling_tables(pdf_path: str) -> list[list[list[object]]]:
    # Disable heavy OCR image rendering on multi-page PDFs to prevent
    # std::bad_alloc out-of-memory crashes.
    opts = PdfPipelineOptions(do_ocr=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(pdf_path)

    tables: list[list[list[object]]] = []
    for table in result.document.tables:
        df = table.export_to_dataframe().fillna("")
        header = [str(c) for c in df.columns]
        body = [list(r) for r in df.itertuples(index=False, name=None)]
        # Docling numbers the columns 0..n when a table has no header row
        # (typically a continuation page). Those placeholders are not data.
        if all(h.strip().isdigit() for h in header):
            tables.append(body)
        else:
            tables.append([list(header), *body])
    return tables


def _pdfplumber_tables(pdf_path: str) -> list[list[list[object]]]:
    tables: list[list[list[object]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if table:
                    tables.append([list(row) for row in table if row])
    return tables


class UnrecognisedStatementError(ValueError):
    """No transactions could be read from the statement's tables."""


def _unrecognised(parsed: list[TableParseResult]) -> UnrecognisedStatementError:
    headers = [h for p in parsed for h in p.unrecognised_headers][:3]
    seen = "; ".join(" | ".join(h) for h in headers) or "no tables found"
    return UnrecognisedStatementError(
        f"No transactions could be extracted from this statement. Table headers seen: {seen}"
    )


def extract_transactions(pdf_path: str) -> list[dict[str, Any]]:
    """Docling first, pdfplumber as fallback — both for a crash and for a
    layout Docling returned but that yielded no transactions.

    Raises UnrecognisedStatementError rather than returning an empty list, so
    an unsupported layout fails the task visibly instead of "completing" with
    zero transactions and no report.
    """
    attempts: list[TableParseResult] = []
    docling_error: Exception | None = None
    try:
        parsed = rows_from_tables(_docling_tables(pdf_path))
        attempts.append(parsed)
        rows = _rows_to_records(parsed)
        if rows:
            return rows
        logger.warning("Docling found no transactions, trying pdfplumber fallback...")
    except Exception as e:
        docling_error = e
        logger.warning(f"Docling extraction failed ({e}), using pdfplumber fallback...")

    try:
        parsed = rows_from_tables(_pdfplumber_tables(pdf_path))
    except Exception as fallback_err:
        logger.error(f"pdfplumber extraction also failed: {fallback_err}")
        # Surface the original Docling failure, but keep the fallback's
        # traceback attached so both causes are visible.
        if docling_error is not None:
            raise docling_error from fallback_err
        raise
    attempts.append(parsed)
    rows = _rows_to_records(parsed)
    if not rows:
        raise _unrecognised(attempts)
    return rows


def process_pdf_task(task_id: str, pdf_path: str) -> None:
    """Background task to process the PDF and update the task store."""
    try:
        update_task_status(task_id, "processing")

        # 1. Extract tables via Docling (with pdfplumber fallback)
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
        payee_targets = [txn.get("payee") or txn.get("particulars", "") for txn in transactions]
        classifications = classifier.classify_batch(payee_targets)
        for txn, cls_result in zip(transactions, classifications, strict=False):
            txn["payee_type"] = cls_result["label"]
            txn["payee_confidence"] = cls_result["confidence"]

        # 4. Enrich merchant payees (Issue #7 / #7b) — cache -> dictionary -> LLM.
        merchant_names = [
            txn.get("payee", "") for txn in transactions if txn.get("payee_type") == "merchant"
        ]
        enrichment_service = get_enrichment_service()
        enrichments = enrichment_service.enrich(merchant_names)
        enrich_iter = iter(enrichments)

        for txn in transactions:
            if txn.get("payee_type") == "merchant":
                e = next(enrich_iter)
                txn["category"] = e.category
                txn["is_essential"] = e.is_essential
                txn["risk_flag"] = e.risk_flag
                txn["lifestyle_dim"] = e.lifestyle_dim
                txn["recurring_type"] = e.recurring_type
            else:
                txn["category"] = None
                txn["is_essential"] = None
                txn["risk_flag"] = None
                txn["lifestyle_dim"] = None
                txn["recurring_type"] = None

        # 5. Score (Track B — Issues #10-#13)
        # Scoring stays best-effort — a statement too thin to score still
        # completes with its transactions — but the reason is logged instead of
        # being swallowed, so "report is null" is diagnosable.
        report = None
        try:
            tx_models = [Transaction(**t) for t in transactions]
            features = build_features(tx_models)
            profile = build_profile(tx_models, features)
            archetype = classify_archetype(features, profile)
            report = score(features, profile, archetype)
        except Exception:
            logger.exception(f"Task {task_id}: scoring skipped, no report produced")

        # 6. Save results to task store
        report_data = report.model_dump(mode="json") if report else None
        update_task_status(
            task_id,
            "completed",
            transactions=transactions,
            report=report_data,
        )

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        update_task_status(task_id, "failed", error=str(e))
