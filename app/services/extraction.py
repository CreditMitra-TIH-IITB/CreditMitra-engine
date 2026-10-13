import logging
import os
import re
import time
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
from app.services.statement_layout import (
    AMOUNT,
    BALANCE,
    DATE,
    DEPOSITS,
    WITHDRAWALS,
    TableParseResult,
    canonical_column,
    rows_from_tables,
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
        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0, "num_predict": 32},
    }
    try:
        resp = client.post(f"{settings.OLLAMA_HOST}/api/generate", json=payload, timeout=60.0)
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()
    except Exception as e:
        logger.error(f"Ollama prediction failed: {e}")
        return ""


_COMPACT_RE = re.compile(r"[^a-z0-9]")
_WORD_RE = re.compile(r"[a-z0-9]+")
# Where the counterparty name sits in common UPI narration formats:
#   UPI/DR/601591718046/SWIGGY LI/UTIB/...   (Canara, SBI)
#   UPI-SWIGGY-swiggy@icici-ICIC0DC0099-...  (HDFC)
_UPI_NAME_RES = (
    re.compile(r"^UPI/(?:DR|CR)/\d+/([^/]+)/", re.IGNORECASE),
    re.compile(r"^UPI-([^-]+)-", re.IGNORECASE),
)


def payee_is_grounded(payee: str, narration: str) -> bool:
    """Every word of the predicted payee must be traceable to the narration.

    The payee model sometimes *invents* a name for a truncated narration —
    "NYKAA ERE" -> "PayViva", "CONNAU GHT" -> "Connaught Hill", a person's name
    -> "Axis Bank". Expansion of a truncation is fine ("JIO RECHA" -> "Jio
    Recharge"), so a word counts as present when its first ~60% (at least 3
    characters) appears in the narration.
    """
    haystack = _COMPACT_RE.sub("", narration.lower())
    words = [w for w in _WORD_RE.findall(payee.lower()) if len(w) >= 3]
    if not words:
        return bool(payee.strip())
    return all(w[: max(3, -(-len(w) * 3 // 5))] in haystack for w in words)


def ground_payee(payee: str, narration: str) -> str:
    """Keep the model's payee if it is grounded; otherwise fall back to the
    name segment of a UPI narration, which is where the name literally is."""
    if not payee or payee_is_grounded(payee, narration):
        return payee
    for pattern in _UPI_NAME_RES:
        m = pattern.match(narration.strip())
        if m and re.search(r"[A-Za-z]", m.group(1)):
            fallback = re.sub(r"\s+", " ", m.group(1)).strip().title()
            logger.debug(f"Ungrounded payee {payee!r}; using narration segment {fallback!r}")
            return fallback
    return payee


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


# A vertical gap this many times the page's usual line spacing separates two
# transactions in a borderless table body (see _borderless_body).
_BLOCK_GAP_FACTOR = 1.6
_SUMMARY_LINE_RE = re.compile(r"\b(opening|closing)\s+balance\b", re.IGNORECASE)


def _borderless_body(page: Any, table: Any) -> list[list[object]]:
    """Rows for a table whose header is boxed but whose body is not.

    pdfplumber finds tables by their ruling lines. Statements like Canara's
    e-passbook draw a box around the header row only, so the table it finds is
    the header alone and every transaction below it is missed. Here the
    header's cell boundaries define the columns, each word below the header is
    placed in the column under which it sits, and lines are grouped into one
    transaction per block — blocks being separated by a wider-than-usual
    vertical gap, since one transaction wraps over several lines (narration,
    then the line with date/amount/balance, then a timestamp).

    Blocks with neither a date nor an amount (an orphan "Chq: <ref>" line, a
    page footer) are dropped rather than glued onto a neighbour.
    """
    header_cells = [c for c in table.rows[0].cells if c]
    bounds = sorted((c[0], c[2]) for c in header_cells)
    if len(bounds) < 3:
        return []
    header: list[object] = [str(v or "").strip() for v in (table.extract() or [[]])[0]]

    def column(word: dict[str, Any]) -> int:
        center = (word["x0"] + word["x1"]) / 2
        return min(
            range(len(bounds)),
            key=lambda i: (
                0
                if bounds[i][0] <= center < bounds[i][1]
                else min(abs(center - bounds[i][0]), abs(center - bounds[i][1]))
            ),
        )

    words = [w for w in page.extract_words() if w["top"] >= table.bbox[3]]
    lines: dict[float, list[dict[str, Any]]] = {}
    for w in words:
        lines.setdefault(round(w["top"]), []).append(w)
    tops = sorted(lines)
    if not tops:
        return []
    gaps = sorted(b - a for a, b in zip(tops, tops[1:], strict=False) if b > a)
    spacing = gaps[len(gaps) // 2] if gaps else 0.0

    def is_summary(top: float) -> bool:
        return bool(_SUMMARY_LINE_RE.search(" ".join(w["text"] for w in lines[top])))

    # An "Opening/Closing Balance" line is always its own block: the first
    # transaction can sit close enough beneath it to be merged otherwise, and
    # the merged row is then dropped as a summary row.
    blocks: list[list[float]] = [[tops[0]]]
    for prev, top in zip(tops, tops[1:], strict=False):
        wide_gap = spacing and top - prev > _BLOCK_GAP_FACTOR * spacing
        if wide_gap or is_summary(prev) or is_summary(top):
            blocks.append([])
        blocks[-1].append(top)

    roles = [canonical_column(h) for h in header]
    date_cols = [i for i, r in enumerate(roles) if r == DATE]
    amount_cols = [i for i, r in enumerate(roles) if r in (DEPOSITS, WITHDRAWALS, AMOUNT, BALANCE)]

    rows: list[list[object]] = [header]
    for block in blocks:
        cells: list[list[str]] = [[] for _ in bounds]
        for top in block:
            for w in sorted(lines[top], key=lambda w: w["x0"]):
                cells[column(w)].append(w["text"])
        row: list[object] = [" ".join(c) for c in cells]
        has_date = any(parse_date(str(row[i])) is not None for i in date_cols if i < len(row))
        has_amount = any(parse_amount(str(row[i])) is not None for i in amount_cols if i < len(row))
        if has_date or has_amount:
            rows.append(row)
    return rows if len(rows) > 1 else []


def _pdfplumber_tables(pdf_path: str) -> list[list[list[object]]]:
    tables: list[list[list[object]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for found in page.find_tables():
                rows: list[list[object]] = [list(row) for row in found.extract() if row]
                if len(rows) == 1:
                    # Header-only: the body is unruled. Rebuild it from words.
                    rows = _borderless_body(page, found) or rows
                if rows:
                    tables.append(rows)
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


def process_pdf_task(task_id: str, pdf_path: str, filename: str | None = None) -> None:
    """Background task to process the PDF and update the task store with live telemetry."""
    try:
        started = time.monotonic()
        clean_filename = filename or os.path.basename(pdf_path)

        # 1. Start extraction
        update_task_status(
            task_id,
            "processing",
            stage="extracting",
            stage_index=1,
            stage_name="PDF Table Extraction (Docling)",
            progress=10.0,
            message="Initializing Docling visual engine and parsing table boundaries...",
            filename=clean_filename,
            started_at=started,
            elapsed_seconds=0.0,
            eta_seconds=25.0,
            processed_rows=0,
            total_rows=0,
        )

        transactions = extract_transactions(pdf_path)
        total_txns = len(transactions)
        extract_elapsed = time.monotonic() - started
        logger.info(
            f"Task {task_id}: extracted {total_txns} rows in {extract_elapsed:.1f}s"
        )

        # 2. Enrich via Ollama (payee-lora) with live row-by-row progress & ETA
        update_task_status(
            task_id,
            "processing",
            stage="payees",
            stage_index=2,
            stage_name="Payee Normalization (payee-lora)",
            progress=25.0,
            message=f"Extracted {total_txns} transaction rows. Beginning payee resolution...",
            total_rows=total_txns,
            processed_rows=0,
            elapsed_seconds=round(time.monotonic() - started, 1),
            eta_seconds=round(max(5.0, total_txns * 0.35), 1),
        )

        payees_started = time.monotonic()
        with httpx.Client(timeout=120.0) as client:
            for idx, txn in enumerate(transactions):
                narration = txn.get("particulars", "")
                if narration and narration not in ("Opening Balance", "Closing Balance"):
                    txn["payee"] = ground_payee(predict_payee(narration, client), narration)
                else:
                    txn["payee"] = ""

                rows_done = idx + 1
                # Update progress every 4 rows or on the very last row, or every row if small
                if rows_done % 4 == 0 or rows_done == total_txns or total_txns <= 12:
                    elapsed = time.monotonic() - started
                    payees_elapsed = time.monotonic() - payees_started
                    rate = rows_done / payees_elapsed if payees_elapsed > 0 else 1.0
                    remaining = total_txns - rows_done
                    eta = max(1.0, remaining / rate) if remaining > 0 else 3.0
                    # Scale progress from 25% to 70% during payee resolution
                    progress_pct = round(25.0 + (rows_done / total_txns) * 45.0, 1)

                    update_task_status(
                        task_id,
                        "processing",
                        stage="payees",
                        stage_index=2,
                        stage_name="Payee Normalization (payee-lora)",
                        progress=min(70.0, progress_pct),
                        message=f"Resolving payees: {rows_done}/{total_txns} rows completed ({rate:.1f} rows/s)",
                        total_rows=total_txns,
                        processed_rows=rows_done,
                        elapsed_seconds=round(elapsed, 1),
                        eta_seconds=round(eta + 4.0, 1),
                        current_narration=narration[:55] if narration else None,
                    )

        empty = sum(1 for t in transactions if not t.get("payee"))
        logger.info(
            f"Task {task_id}: payees predicted at {time.monotonic() - started:.0f}s ({empty} empty)"
        )

        # 3. Classify payees as person/merchant
        update_task_status(
            task_id,
            "processing",
            stage="classification",
            stage_index=3,
            stage_name="Entity Classification (ONNX Qwen3)",
            progress=72.0,
            message="Classifying entities as individuals (P2P) vs commercial merchants (P2M)...",
            total_rows=total_txns,
            processed_rows=total_txns,
            elapsed_seconds=round(time.monotonic() - started, 1),
            eta_seconds=3.0,
        )

        from app.services.merchant_classifier import MerchantClassifierService

        classifier = MerchantClassifierService.get_instance()
        payee_targets = [txn.get("payee") or txn.get("particulars", "") for txn in transactions]
        unique_targets = list(dict.fromkeys(payee_targets))
        by_target = dict(
            zip(unique_targets, classifier.classify_batch(unique_targets), strict=True)
        )
        for txn, target in zip(transactions, payee_targets, strict=True):
            cls_result = by_target[target]
            txn["payee_type"] = cls_result["label"]
            txn["payee_confidence"] = cls_result["confidence"]

        # 4. Enrich merchant payees (Issue #7 / #7b)
        update_task_status(
            task_id,
            "processing",
            stage="enrichment",
            stage_index=3,
            stage_name="Merchant Taxonomy & Risk Tagging",
            progress=85.0,
            message="Evaluating expense taxonomies, essentiality, and financial risk tags...",
            total_rows=total_txns,
            processed_rows=total_txns,
            elapsed_seconds=round(time.monotonic() - started, 1),
            eta_seconds=2.0,
        )

        merchant_names = [
            txn.get("payee", "") for txn in transactions if txn.get("payee_type") == "merchant"
        ]
        enrichment_service = get_enrichment_service()
        statement_id = None
        if settings.MERCHANT_ENRICHMENT_URL:
            from app.services.merchant_enrichment_client import statement_fingerprint

            with open(pdf_path, "rb") as fh:
                statement_id = statement_fingerprint(fh.read())
        enrichments = enrichment_service.enrich(merchant_names, statement_id=statement_id)
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

        # 5. Score (Track B)
        update_task_status(
            task_id,
            "processing",
            stage="scoring",
            stage_index=4,
            stage_name="Alternative Lifestyle Credit Scoring",
            progress=92.0,
            message="Computing L1-L6 lifestyle dimensions and 300-900 alternative credit score...",
            total_rows=total_txns,
            processed_rows=total_txns,
            elapsed_seconds=round(time.monotonic() - started, 1),
            eta_seconds=1.0,
        )

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
        total_time = round(time.monotonic() - started, 1)
        report_data = report.model_dump(mode="json") if report else None
        update_task_status(
            task_id,
            "completed",
            stage="completed",
            stage_index=4,
            stage_name="Underwriting Complete",
            progress=100.0,
            message=f"Successfully extracted and scored {total_txns} transactions in {total_time}s",
            total_rows=total_txns,
            processed_rows=total_txns,
            elapsed_seconds=total_time,
            eta_seconds=0.0,
            transactions=transactions,
            report=report_data,
        )
        logger.info(
            f"Task {task_id}: completed in {total_time:.1f}s,"
            f" report={'yes' if report else 'none'}"
        )

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        update_task_status(task_id, "failed", error=str(e))

    finally:
        # The upload is a user's bank statement. Once extracted, nothing reads
        # it again, so it must not outlive the task — success or failure.
        try:
            os.remove(pdf_path)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.error(f"Task {task_id}: could not delete upload {pdf_path}: {e}")
