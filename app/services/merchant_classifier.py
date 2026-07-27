"""Merchant Classifier Service
============================
ONNX-based merchant/person classification for payee names with robust
dictionary + heuristic fallback.

No hard PyTorch dependency — uses onnxruntime + transformers (tokenizer only)
when available, with intelligent fallback to dictionary & rule-based regex
classification so payee classification NEVER fails or skips.

Singleton pattern: loads models once, reused across all requests.
"""

import logging
import os
import re
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Resolve model paths relative to project data directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS_DIR = os.path.join(_BASE_DIR, "data", "models")
_ONNX_DIR = os.path.join(_MODELS_DIR, "onnx_pipeline")

# Regex for common Indian merchant / enterprise / business indicators
_BUSINESS_KEYWORDS_RE = re.compile(
    r"\b("
    r"LTD|LIMITED|PVT|INC|CORP|ENTERPRISE|ENTERPRISES|EN|STORES|STORE|SERVICES|RECHARGE|"
    r"INFOTECH|SOLUTIONS|LLP|AGENCY|AGENCIES|MART|RETAIL|BILLS|PAY|INSTAMART|PAYTM|RAZORPAY|"
    r"CASHFREE|P2M|FOOD|TRADERS|TRADING|HOTEL|CAFE|RESTAURANT|CLINIC|HOSPITAL|PHARMA|PHARMACY|"
    r"BAKERY|SUPERMARKET|MARKET|JEWELLERS|FUELS|PETROL|STATION|BROKING|CLEARING|INSURANCE|"
    r"ELECTRONICS|FASHION|SILK|FOOTWEAR|ONLINE|CREATION|CREATIONS|STUDIO|MOTORS|AUTO|MOBIKWIK|"
    r"GPAY|PHONEPE|BHARATPE|FLIPKART|SWIGGY|ZOMATO|AMAZON|BLINKIT|ZEPTO|BIGBASKET|JIO|AIRTEL|"
    r"IRCTC|UBER|OLA|NETFLIX|GROWW|ZERODHA|LIC|SIMPL|LAZYPAY|DREAM11|WAZIRX|DMART|MYNTRA|MEESHO|"
    r"AJIO|NYKAA|BSNL|HPCL|BPCL|IOCL|SHELL|GOIBIBO|MAKEMYTRIP|INDIAN OIL|DELHIVERY"
    r")\b",
    re.IGNORECASE,
)


def heuristic_classify_payee(name: str) -> dict[str, Any]:
    """Fallback classification when ONNX models are unavailable or throw errors.

    Returns dict with label ('merchant' | 'person'), confidence, p_merchant.
    """
    if not name or not name.strip():
        return {"label": None, "confidence": None, "p_merchant": None}

    raw = name.strip()

    # 1. Check merchant dictionary
    try:
        from app.services.merchant_enrichment import get_enrichment_service, normalize_name

        norm = normalize_name(raw)
        dict_hit = get_enrichment_service()._dictionary.lookup(norm)
        if dict_hit is not None:
            return {"label": "merchant", "confidence": 0.95, "p_merchant": 0.95}
    except Exception:
        pass

    # 2. Check business keywords / regex
    if _BUSINESS_KEYWORDS_RE.search(raw):
        return {"label": "merchant", "confidence": 0.88, "p_merchant": 0.88}

    # 3. Check domain names, digits, or UPI VPA handles
    if any(char in raw for char in ["@", ".com", ".in", "http", "/", "_"]):
        return {"label": "merchant", "confidence": 0.82, "p_merchant": 0.82}

    # 4. Standard personal name heuristic (1-3 words, no numbers/business suffixes)
    words = raw.split()
    if 1 <= len(words) <= 3 and all(w.isalpha() for w in words):
        return {"label": "person", "confidence": 0.80, "p_merchant": 0.20}

    # Default to person with reasonable confidence
    return {"label": "person", "confidence": 0.70, "p_merchant": 0.30}


class MerchantClassifierService:
    """Singleton merchant classifier.

    Classifies payee names as 'person' or 'merchant' using ONNX embeddings + AttentionMLP
    when available, or dictionary + heuristic fallback.
    """

    _instance: "MerchantClassifierService | None" = None
    _initialized: bool = False

    @classmethod
    def get_instance(cls) -> "MerchantClassifierService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        if MerchantClassifierService._initialized:
            return

        self._available = False

        try:
            self._load_models()
            self._available = True
            MerchantClassifierService._initialized = True
            logger.info("MerchantClassifierService initialized ONNX model successfully")
        except Exception as e:
            logger.warning(
                f"ONNX merchant classifier unavailable ({e}). Using dictionary + heuristic fallback."
            )

    def _load_models(self) -> None:
        embedding_path = os.path.join(_ONNX_DIR, "qwen3_embedding.onnx")
        classifier_path = os.path.join(_MODELS_DIR, "attentionmlp.onnx")
        scaler_path = os.path.join(_MODELS_DIR, "scaler.joblib")
        tokenizer_dir = os.path.join(_ONNX_DIR, "tokenizer")

        for path in [embedding_path, classifier_path, scaler_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required model file not found: {path}")

        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

        from transformers import AutoTokenizer
        import onnxruntime as ort

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        self._embedding_session = ort.InferenceSession(
            embedding_path, providers=["CPUExecutionProvider"]
        )
        self._classifier_session = ort.InferenceSession(
            classifier_path, providers=["CPUExecutionProvider"]
        )
        self._scaler = joblib.load(scaler_path)

    @property
    def available(self) -> bool:
        # Always available because heuristic fallback is active!
        return True

    def _embed(self, name: str) -> np.ndarray[Any, Any]:
        inputs = self._tokenizer(
            name,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        embedding: np.ndarray[Any, Any] = self._embedding_session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )[0]
        return embedding

    def classify(self, name: str) -> dict[str, Any]:
        if not name or not name.strip():
            return {"label": None, "confidence": None, "p_merchant": None}
        results = self.classify_batch([name])
        return results[0]

    def classify_batch(self, names: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for name in names:
            if not name or not name.strip():
                results.append({"label": None, "confidence": None, "p_merchant": None})
                continue

            # If ONNX model is loaded, try ONNX inference first
            if self._available:
                try:
                    embedding = self._embed(name)
                    scaled = self._scaler.transform(embedding).astype(np.float32)
                    logit = self._classifier_session.run(None, {"embedding": scaled})[0]
                    p_merchant = float(1.0 / (1.0 + np.exp(-logit.item())))

                    is_merchant = p_merchant >= 0.5
                    confidence = p_merchant if is_merchant else (1.0 - p_merchant)

                    results.append(
                        {
                            "label": "merchant" if is_merchant else "person",
                            "confidence": round(confidence, 4),
                            "p_merchant": round(p_merchant, 4),
                        }
                    )
                    continue
                except Exception as e:
                    logger.debug(f"ONNX classification failed for '{name}': {e}. Using heuristic.")

            # Heuristic fallback (dictionary + regex)
            results.append(heuristic_classify_payee(name))

        return results
