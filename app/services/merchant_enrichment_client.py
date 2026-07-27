"""HTTP Client for Standalone Merchant Enrichment Server (Issue #9).

Connects to the external merchant-server via HTTP (`MERCHANT_ENRICHMENT_URL`),
falling back gracefully to local dictionary lookup if unconfigured, offline, or
errored.
"""

from __future__ import annotations

import logging
import httpx
from app.core.config import settings
from app.schemas.statements import MerchantEnrichment

logger = logging.getLogger(__name__)

def enrich_merchants_via_http(merchant_names: list[str]) -> list[MerchantEnrichment] | None:
    if not settings.MERCHANT_ENRICHMENT_URL or not merchant_names:
        return None

    url = f"{settings.MERCHANT_ENRICHMENT_URL.rstrip('/')}/api/v1/merchants/enrich"
    headers = {"Content-Type": "application/json"}
    if settings.MERCHANT_API_KEY:
        headers["X-API-Key"] = settings.MERCHANT_API_KEY

    payload = {"merchants": merchant_names}

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return [MerchantEnrichment(**item) for item in results]
    except Exception as e:
        logger.warning(f"HTTP merchant enrichment failed ({url}): {e}. Falling back to local resolver.")
        return None
