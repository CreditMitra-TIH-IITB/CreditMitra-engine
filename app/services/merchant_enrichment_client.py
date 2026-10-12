"""HTTP Client for Standalone Merchant Enrichment Server (Issue #9).

Connects to the external merchant-server via HTTP (`MERCHANT_ENRICHMENT_URL`),
falling back gracefully to local dictionary lookup if unconfigured, offline, or
errored.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from pathlib import Path

import httpx

from app.core.config import settings
from app.schemas.statements import MerchantEnrichment

logger = logging.getLogger(__name__)


def _statement_secret() -> bytes:
    """A random per-install key, created on first use, that never leaves the
    device. Keying the fingerprint with it means the server cannot confirm a
    guess about which statement a fingerprint came from."""
    path = Path(settings.DATA_DIR) / ".statement_secret"
    try:
        return path.read_bytes()
    except FileNotFoundError:
        secret = secrets.token_bytes(32)
        path.write_bytes(secret)
        return secret


def statement_fingerprint(pdf_bytes: bytes) -> str:
    """Opaque, stable id for a statement: HMAC-SHA256 of the file under the
    install secret. The same statement uploaded twice gets the same id, which
    is what lets the merchant server count distinct statements per unresolved
    name (its privacy gate) instead of requests."""
    return hmac.new(_statement_secret(), pdf_bytes, hashlib.sha256).hexdigest()


def enrich_merchants_via_http(
    merchant_names: list[str], statement_id: str | None = None
) -> list[MerchantEnrichment] | None:
    if not settings.MERCHANT_ENRICHMENT_URL or not merchant_names:
        return None

    url = f"{settings.MERCHANT_ENRICHMENT_URL.rstrip('/')}/api/v1/merchants/enrich"
    headers = {"Content-Type": "application/json"}
    if settings.MERCHANT_API_KEY:
        headers["X-API-Key"] = settings.MERCHANT_API_KEY

    payload: dict[str, object] = {"merchants": merchant_names}
    if statement_id:
        payload["statement_id"] = statement_id

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return [MerchantEnrichment(**item) for item in results]
    except Exception as e:
        logger.warning(
            f"HTTP merchant enrichment failed ({url}): {e}. Falling back to local resolver."
        )
        return None
