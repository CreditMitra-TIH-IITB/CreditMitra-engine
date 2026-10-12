"""Dictionary matching: coverage of the curated entries, and the guard that keeps
ordinary names from prefix-matching into a high-severity category."""

from __future__ import annotations

import json

import pytest

from app.services.merchant_enrichment import (
    _CATEGORY_TAXONOMY,
    _DICT_PATH,
    _HIGH_SEVERITY_CATEGORIES,
    _Dictionary,
    normalize_name,
)


@pytest.fixture(scope="module")
def dictionary() -> _Dictionary:
    return _Dictionary()


def _category(dictionary: _Dictionary, raw: str) -> str | None:
    hit = dictionary.lookup(normalize_name(raw))
    return hit.category if hit else None


@pytest.mark.parametrize(
    "raw",
    [
        "Navin Kumar",  # Navi -> bnpl_lending
        "SIMPLE MART",  # Simpl -> bnpl_lending
        "Slice of Italy",  # slice -> bnpl_lending
        "KUBER GENERAL STORE",  # CoinSwitch Kuber -> crypto
        "CASHEW TRADERS",  # CASHe -> bnpl_lending
        "FIBER WORLD",  # Fibe -> bnpl_lending
        "Dreamland Bakery",  # Dream11 -> gambling
    ],
)
def test_ordinary_names_never_reach_a_high_severity_category(
    dictionary: _Dictionary, raw: str
) -> None:
    assert _category(dictionary, raw) not in _HIGH_SEVERITY_CATEGORIES


@pytest.mark.parametrize(
    ("raw", "category"),
    [
        ("NAVI", "bnpl_lending"),  # exact
        ("KREDITBE", "bnpl_lending"),  # UPI truncation of an alias
        ("DREAM11", "gambling"),
        ("SIMPLILEARN", "education"),  # must not fall into Simpl
        ("SWIGGY LI", "food_delivery"),
        ("FlipKaar", "shopping"),  # payee-model spelling seen in real runs
        ("CONNAUGHT PLAZA RESTAURANTS PVT LTD", "dining"),  # McDonald's franchisee
        ("DOMINOS PIZZA", "dining"),
        ("BESCOM", "utilities"),
        ("BAJAJ FINANCE LTD", "loan_emi"),
        ("PAYTM MONEY", "investments"),  # not swallowed by the generic Paytm entry
    ],
)
def test_known_merchants_resolve(dictionary: _Dictionary, raw: str, category: str) -> None:
    assert _category(dictionary, raw) == category


def test_every_entry_agrees_with_the_taxonomy() -> None:
    """is_essential / lifestyle_dim / risk_flag are a function of category."""
    merchants = json.loads(_DICT_PATH.read_text(encoding="utf-8"))["merchants"]
    for m in merchants:
        assert m["category"] in _CATEGORY_TAXONOMY, m["canonical_name"]
        expected = _CATEGORY_TAXONOMY[m["category"]]
        actual = (m["is_essential"], m["lifestyle_dim"], m["risk_flag"])
        assert actual == expected, m["canonical_name"]
