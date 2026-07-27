"""Issue #9 — Privacy assertion test.

Asserts that outbound payloads to the standalone merchant-server contain
ONLY merchant name strings — no transaction narrations, amounts, dates, balances,
or personal payees cross the boundary.
"""

from __future__ import annotations

import httpx
import pytest
from unittest.mock import MagicMock, patch
from app.services.merchant_enrichment_client import enrich_merchants_via_http

def test_privacy_payload_contains_only_merchant_name_strings():
    sample_names = ["SWIGGY LI", "JIO RECHA", "IRCTCT OUR"]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "canonical_name": "Swiggy",
                "category": "food_delivery",
                "is_essential": False,
                "risk_flag": None,
                "lifestyle_dim": "aspirational",
                "recurring_type": "adhoc",
            },
            {
                "canonical_name": "Jio",
                "category": "telecom",
                "is_essential": True,
                "risk_flag": None,
                "lifestyle_dim": "essential",
                "recurring_type": "subscription",
            },
            {
                "canonical_name": "IRCTC",
                "category": "transport",
                "is_essential": True,
                "risk_flag": None,
                "lifestyle_dim": "essential",
                "recurring_type": "adhoc",
            },
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        with patch("app.core.config.settings.MERCHANT_ENRICHMENT_URL", "http://localhost:8001"):
            with patch("app.core.config.settings.MERCHANT_API_KEY", "test-key"):
                res = enrich_merchants_via_http(sample_names)

                assert res is not None
                assert len(res) == 3

                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                payload = kwargs.get("json", {})

                # Privacy assertions:
                assert "merchants" in payload
                assert isinstance(payload["merchants"], list)
                assert payload["merchants"] == sample_names

                # Explicitly verify NO transaction metadata keys exist
                forbidden_keys = {"date", "narration", "particulars", "amount", "balance", "deposits", "withdrawals", "payee"}
                assert forbidden_keys.isdisjoint(payload.keys())
