"""The on-device vector tier: near misses of dictionary merchants resolve, while
local shops, people, and anything high-severity never do."""

from __future__ import annotations

import pytest

from app.services.merchant_enrichment import MerchantEnrichmentService, normalize_name


@pytest.fixture(scope="module")
def service() -> MerchantEnrichmentService:
    return MerchantEnrichmentService()


def _local(service: MerchantEnrichmentService, raw: str) -> tuple[str, str] | None:
    hit = service._lookup_local(normalize_name(raw))
    return (hit[0].canonical_name, hit[1]) if hit else None


@pytest.mark.parametrize(
    ("raw", "merchant"),
    [
        ("SWIGY", "Swiggy"),  # dropped letter
        ("BIGBASKT", "BigBasket"),
        ("NETFLX", "Netflix"),
        ("CONNAU GHT PLAZA", "McDonald's"),  # truncation split mid-word
    ],
)
def test_near_misses_resolve_through_the_vector_tier(
    service: MerchantEnrichmentService, raw: str, merchant: str
) -> None:
    assert _local(service, raw) == (merchant, "vector")


@pytest.mark.parametrize("raw", ["SHINY CH", "K K CATER", "RAMESH KUMAR", "AMAN ENTE"])
def test_local_shops_and_people_do_not_match(service: MerchantEnrichmentService, raw: str) -> None:
    assert _local(service, raw) is None


@pytest.mark.parametrize("raw", ["DREEM11", "KREDITBEEE", "WAZIRXX"])
def test_high_severity_merchants_are_never_fuzzy_matched(
    service: MerchantEnrichmentService, raw: str
) -> None:
    assert _local(service, raw) is None


def test_exact_dictionary_hits_are_still_labelled_dictionary(
    service: MerchantEnrichmentService,
) -> None:
    assert _local(service, "SWIGGY") == ("Swiggy", "dictionary")


def test_vector_tier_can_be_switched_off(
    service: MerchantEnrichmentService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.core.config.settings.ENRICHMENT_VECTOR_ENABLED", False)
    assert _local(service, "SWIGY") is None
