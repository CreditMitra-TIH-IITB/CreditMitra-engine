"""
Merchant Classifier Service
============================
ONNX-based merchant/person classification for payee names.
No PyTorch dependency — uses onnxruntime + transformers (tokenizer only).

Singleton pattern: loads models once, reused across all requests.
"""

import logging
import os
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Resolve model paths relative to project data directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS_DIR = os.path.join(_BASE_DIR, "data", "models")
_ONNX_DIR = os.path.join(_MODELS_DIR, "onnx_pipeline")


class MerchantClassifierService:
    """Singleton ONNX merchant classifier.

    Classifies payee names as 'person' or 'merchant' using:
    1. Qwen3-Embedding-0.6B (ONNX) for text → 1024-dim embedding
    2. StandardScaler for feature normalization
    3. AttentionMLP (ONNX) for classification

    Usage:
        clf = MerchantClassifierService.get_instance()
        result = clf.classify("Swiggy Instamart")
        # → {"label": "merchant", "confidence": 0.9797}
    """

    _instance: "MerchantClassifierService | None" = None
    _initialized: bool = False

    @classmethod
    def get_instance(cls) -> "MerchantClassifierService":
        """Get or create the singleton classifier instance."""
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
            logger.info("MerchantClassifierService initialized successfully")
        except FileNotFoundError as e:
            logger.warning(f"Merchant classifier models not found: {e}")
            logger.warning("Classification will be skipped. Run export scripts to generate models.")
        except Exception as e:
            logger.error(f"Failed to initialize merchant classifier: {e}")

    def _load_models(self) -> None:
        """Load ONNX models, tokenizer, and scaler."""
        # Check required files exist
        embedding_path = os.path.join(_ONNX_DIR, "qwen3_embedding.onnx")
        classifier_path = os.path.join(_MODELS_DIR, "attentionmlp.onnx")
        scaler_path = os.path.join(_MODELS_DIR, "scaler.joblib")
        tokenizer_dir = os.path.join(_ONNX_DIR, "tokenizer")

        for path in [embedding_path, classifier_path, scaler_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required model file not found: {path}")

        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

        # Load tokenizer
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

        # Load ONNX sessions
        import onnxruntime as ort

        self._embedding_session = ort.InferenceSession(
            embedding_path, providers=["CPUExecutionProvider"]
        )
        self._classifier_session = ort.InferenceSession(
            classifier_path, providers=["CPUExecutionProvider"]
        )

        # Load scaler
        self._scaler = joblib.load(scaler_path)

        logger.info(
            "Loaded ONNX models: embedding=%s, classifier=%s",
            embedding_path,
            classifier_path,
        )

    @property
    def available(self) -> bool:
        """Whether the classifier is ready to use."""
        return self._available

    def _embed(self, name: str) -> np.ndarray[Any, Any]:
        """Generate L2-normalized embedding for a single name."""
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
        return embedding  # (1, 1024), already L2-normalized

    def classify(self, name: str) -> dict[str, Any]:
        """Classify a single payee name.

        Args:
            name: The payee name string.

        Returns:
            dict with: label, confidence, p_merchant
            Returns {"label": None} if classifier is not available.
        """
        if not self._available or not name or not name.strip():
            return {"label": None, "confidence": None, "p_merchant": None}

        results = self.classify_batch([name])
        return results[0]

    def classify_batch(self, names: list[str]) -> list[dict[str, Any]]:
        """Classify a batch of payee names.

        Args:
            names: List of payee name strings.

        Returns:
            List of dicts with: label, confidence, p_merchant
        """
        if not self._available:
            return [{"label": None, "confidence": None, "p_merchant": None}] * len(names)

        results: list[dict[str, Any]] = []

        for name in names:
            if not name or not name.strip():
                results.append({"label": None, "confidence": None, "p_merchant": None})
                continue

            try:
                # 1. Embed
                embedding = self._embed(name)
                # 2. Scale
                scaled = self._scaler.transform(embedding).astype(np.float32)
                # 3. Classify
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
            except Exception as e:
                logger.error(f"Classification failed for '{name}': {e}")
                results.append({"label": None, "confidence": None, "p_merchant": None})

        return results
