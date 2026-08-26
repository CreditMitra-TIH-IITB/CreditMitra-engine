"""Tier ordering between the on-device resolver and the shared merchant server.

The server is a *tier*, not a replacement. Getting this wrong is not a
performance detail: the previous implementation handed every name to the server
and returned its answers wholesale, so switching the server on swapped an
85-merchant local dictionary plus the cache and the LLM tier for whatever the
server happened to know. Merchants that resolved fine offline stopped resolving
the moment MERCHANT_ENRICHMENT_URL was set.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import app.services.merchant_enrichment as enrichment_module
from app.schemas.statements import MerchantEnrichment
from app.services.merchant_enrichment import MerchantEnrichmentService

DICTIONARY = {
    "_meta": {"note": "test fixture"},
    "merchants": [
        {
            "canonical_name": "Swiggy",
            "aliases": ["swiggy", "swiggy li"],
            "category": "food_delivery",
            "is_essential": False,
            "risk_flag": None,
            "lifestyle_dim": "aspirational",
            "recurring_type": "adhoc",
        }
    ],
    "gig_payout_sources": {"aliases": ["swiggy"]},
}

NYKAA = {
    "canonical_name": "Nykaa",
    "aliases": ["nykaa"],
    "category": "personal_care",
    "is_essential": False,
    "risk_flag": None,
    "lifestyle_dim": "aspirational",
    "recurring_type": "adhoc",
}


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A service with its own dictionary and cache, and no LLM tier."""
    dict_path = tmp_path / "india_merchants.json"
    dict_path.write_text(json.dumps(DICTIONARY), encoding="utf-8")
    monkeypatch.setattr(enrichment_module, "_DICT_PATH", dict_path)
    monkeypatch.setattr(enrichment_module, "_CACHE_PATH", tmp_path / "cache.db")
    monkeypatch.setattr(enrichment_module.settings, "ENRICHMENT_LLM_ENABLED", False)
    monkeypatch.setattr(enrichment_module.settings, "MERCHANT_ENRICHMENT_URL", "http://server")
    monkeypatch.setattr(enrichment_module.settings, "MERCHANT_API_KEY", "test-key")
    return MerchantEnrichmentService(), dict_path


def remote(**overrides):
    payload = {
        "canonical_name": "Remote Merchant",
        "category": "shopping",
        "is_essential": False,
        "risk_flag": None,
        "lifestyle_dim": "aspirational",
        "recurring_type": "adhoc",
    }
    payload.update(overrides)
    return MerchantEnrichment(**payload)


PATCH_TARGET = "app.services.merchant_enrichment_client.enrich_merchants_via_http"


def test_locally_known_merchants_are_never_sent_to_the_server(service):
    svc, _ = service
    with patch(PATCH_TARGET, return_value=[]) as http:
        results = svc.enrich(["SWIGGY LI"])

    http.assert_not_called()
    assert results[0].canonical_name == "Swiggy"
    assert results[0].category == "food_delivery"


def test_only_unresolved_names_leave_the_device(service):
    """Fewer names crossing the boundary is both a better result and a smaller
    disclosure."""
    svc, _ = service
    with patch(PATCH_TARGET, return_value=[remote()]) as http:
        svc.enrich(["SWIGGY LI", "MYSTERY STORE"])

    sent = http.call_args.args[0]
    assert sent == ["MYSTERY STORE"]


def test_server_result_fills_a_gap_the_dictionary_cannot(service):
    svc, _ = service
    with patch(
        PATCH_TARGET, return_value=[remote(canonical_name="Nykaa", category="personal_care")]
    ):
        results = svc.enrich(["SWIGGY LI", "NYKAA FASHION"])

    assert [r.category for r in results] == ["food_delivery", "personal_care"]


def test_local_dictionary_wins_over_the_server(service):
    """The regression in one assertion: a merchant the engine knows must keep
    resolving correctly even if the server would answer differently."""
    svc, _ = service
    with patch(PATCH_TARGET, return_value=[remote(canonical_name="WRONG", category="other")]):
        results = svc.enrich(["SWIGGY LI"])

    assert results[0].canonical_name == "Swiggy"


def test_an_unresolved_server_answer_is_not_treated_as_resolved(service):
    """The server returns a well-formed MerchantEnrichment for names it cannot
    resolve too. Accepting `other` as an answer would mark a failure as a
    success and inflate merchant_resolution_rate, which feeds L3."""
    svc, _ = service
    with patch(PATCH_TARGET, return_value=[remote(category="other", lifestyle_dim="neutral")]):
        results = svc.enrich(["MYSTERY STORE"])

    assert results[0].category == "other"


def test_server_outage_falls_back_to_local_tiers(service):
    svc, _ = service
    with patch(PATCH_TARGET, return_value=None):
        results = svc.enrich(["SWIGGY LI", "MYSTERY STORE"])

    assert results[0].category == "food_delivery"
    assert results[1].category == "other"


def test_order_and_duplicates_are_preserved(service):
    svc, _ = service
    names = ["SWIGGY LI", "MYSTERY STORE", "SWIGGY LI"]
    with patch(PATCH_TARGET, return_value=[remote(category="shopping")]):
        results = svc.enrich(names)

    assert len(results) == 3
    assert results[0].category == results[2].category == "food_delivery"


def test_a_short_server_response_is_ignored_rather_than_misaligned(service):
    """Zipping a truncated response against the request would attach one
    merchant's enrichment to another merchant's transactions."""
    svc, _ = service
    with patch(PATCH_TARGET, return_value=[remote(), remote()]):
        results = svc.enrich(["MYSTERY STORE"])

    assert results[0].category == "other"


def test_adding_a_merchant_fixes_a_previously_cached_miss(service):
    """Same defect the server had. The cache is read before the dictionary, so
    without a version stamp a name that missed once stayed missed forever and
    curating the dictionary achieved nothing."""
    svc, dict_path = service
    with patch(PATCH_TARGET, return_value=None):
        assert svc.enrich(["NYKAA"])[0].category == "other"

    data = json.loads(dict_path.read_text(encoding="utf-8"))
    data["merchants"].append(NYKAA)
    dict_path.write_text(json.dumps(data), encoding="utf-8")

    reloaded = MerchantEnrichmentService()
    with patch(PATCH_TARGET, return_value=None):
        assert reloaded.enrich(["NYKAA"])[0].category == "personal_care"


def test_dictionary_version_tracks_content(service):
    svc, dict_path = service
    before = svc._dictionary.version

    data = json.loads(dict_path.read_text(encoding="utf-8"))
    data["merchants"].append(NYKAA)
    dict_path.write_text(json.dumps(data), encoding="utf-8")

    assert MerchantEnrichmentService()._dictionary.version != before
