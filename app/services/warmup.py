"""Background warm-up of the slow-to-load models.

Both cold loads land on the first statement otherwise: Ollama loading the payee
model (~45s for the unquantized build) and the ONNX classifier's 2.3GB embedding
model (~75s). Neither is needed to serve /health or accept an upload, so they
load on a daemon thread and startup stays fast. Every failure is logged and
swallowed — a missing Ollama or model file must never stop the engine booting;
the pipeline already degrades gracefully without them.
"""

from __future__ import annotations

import logging
import threading

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


def _warm_payee_model() -> None:
    # An empty prompt makes Ollama load the model without generating anything.
    try:
        resp = httpx.post(
            f"{settings.OLLAMA_HOST}/api/generate",
            json={"model": settings.OLLAMA_MODEL, "keep_alive": settings.OLLAMA_KEEP_ALIVE},
            timeout=180.0,
        )
        resp.raise_for_status()
        logger.info("Warm-up: payee model %s loaded", settings.OLLAMA_MODEL)
    except Exception as e:
        logger.warning(f"Warm-up: payee model not loaded ({e}); first statement will load it")


def _warm_classifier() -> None:
    try:
        from app.services.merchant_classifier import MerchantClassifierService

        MerchantClassifierService.get_instance()
        logger.info("Warm-up: merchant classifier ready")
    except Exception as e:
        logger.warning(f"Warm-up: merchant classifier not loaded ({e})")


def _warm_enrichment() -> None:
    try:
        from app.services.merchant_enrichment import get_enrichment_service

        get_enrichment_service()
    except Exception as e:
        logger.warning(f"Warm-up: merchant enrichment not loaded ({e})")


def start_warmup() -> None:
    """Kick off warm-up without blocking startup."""
    if not settings.WARMUP_ON_STARTUP:
        return
    for target in (_warm_payee_model, _warm_classifier, _warm_enrichment):
        threading.Thread(target=target, name=f"warmup{target.__name__}", daemon=True).start()
