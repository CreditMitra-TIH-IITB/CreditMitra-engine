"""Issue #9 — Privacy assertion test.

Asserts that outbound payloads to the standalone merchant-server contain
ONLY merchant name strings — no transaction narrations, amounts, dates, balances,
or personal payees cross the boundary — plus, optionally, an opaque statement
fingerprint that cannot be tied back to the statement without a secret that
stays on the device.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.merchant_enrichment_client import (
    enrich_merchants_via_http,
    statement_fingerprint,
)


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

    with (
        patch("httpx.Client.post", return_value=mock_resp) as mock_post,
        patch("app.core.config.settings.MERCHANT_ENRICHMENT_URL", "http://localhost:8001"),
        patch("app.core.config.settings.MERCHANT_API_KEY", "test-key"),
    ):
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
        forbidden_keys = {
            "date",
            "narration",
            "particulars",
            "amount",
            "balance",
            "deposits",
            "withdrawals",
            "payee",
        }
        assert forbidden_keys.isdisjoint(payload.keys())


def _ok_response(n: int) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {
        "results": [
            {
                "canonical_name": "X",
                "category": "other",
                "is_essential": False,
                "risk_flag": None,
                "lifestyle_dim": "neutral",
                "recurring_type": "adhoc",
            }
        ]
        * n
    }
    return resp


def test_payload_keys_are_exactly_merchants_and_an_opaque_statement_id() -> None:
    fingerprint = statement_fingerprint(b"%PDF statement bytes")
    with (
        patch("httpx.Client.post", return_value=_ok_response(1)) as mock_post,
        patch("app.core.config.settings.MERCHANT_ENRICHMENT_URL", "http://localhost:8001"),
    ):
        enrich_merchants_via_http(["SWIGGY LI"], statement_id=fingerprint)

    payload = mock_post.call_args.kwargs["json"]
    assert set(payload) == {"merchants", "statement_id"}
    assert re.fullmatch(r"[0-9a-f]{64}", payload["statement_id"])


def test_fingerprint_is_stable_and_keyed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
    pdf = b"%PDF statement bytes"

    first = statement_fingerprint(pdf)
    assert statement_fingerprint(pdf) == first  # a resubmission counts once
    assert statement_fingerprint(b"%PDF another statement") != first

    # Not a plain hash of the file: the server cannot check a guess without
    # the install secret, which never leaves the device.
    assert first != hashlib.sha256(pdf).hexdigest()
    secret = (tmp_path / ".statement_secret").read_bytes()
    assert first == hmac.new(secret, pdf, hashlib.sha256).hexdigest()
