"""Regression test: the LangChain fallback must classify from real,
web-search-grounded content only — there is no local "guess from the bare
name" step anymore.

Real-world trigger: "Westside" is an Indian fashion & lifestyle retailer
(Trent Ltd / Tata Group), not in data/india_merchants.json. The small local
model (qwen2.5:1.5b-instruct), asked to describe it from the name alone with
no grounding, confidently guessed category "groceries" instead of admitting
it didn't know. That blind-guess step has been removed entirely
(app/services/merchant_enrichment.py `_LLMFallback.enrich`): now the
fallback does nothing at all unless web search is enabled and returns real
content to ground the classification in.

Uses a fake web-search object, not a live LangSearch call — deterministic,
no API key required in CI.
"""

from __future__ import annotations

from app.schemas.statements import MerchantEnrichment
from app.services.merchant_enrichment import _LLMFallback


class _RecordingRunnable:
    def __init__(self, results):
        self.calls: list[dict] = []
        self._results = iter(results)

    def invoke(self, payload):
        self.calls.append(payload)
        return next(self._results)


class _FakeWebSearch:
    """Stands in for the real LangSearch-backed _WebSearch tier."""

    def __init__(self, result: str | None, enabled: bool = True):
        self.enabled = enabled
        self.queries: list[str] = []
        self._result = result

    def search(self, name: str) -> str | None:
        self.queries.append(name)
        return self._result


def test_classifies_from_grounded_web_content():
    """Westside case: web search returns the real description, the
    categorize step produces the correct taxonomy-consistent result."""
    fallback = _LLMFallback()
    fallback._failed = False
    fallback._search = _FakeWebSearch(
        "Westside — fashion and lifestyle retail chain in India, part of Trent Ltd (Tata Group)."
    )

    grounded = {
        "summary": (
            "Westside is an Indian fashion and lifestyle retail chain owned by "
            "Trent Ltd (Tata Group), selling clothing, footwear, and home decor."
        )
    }
    summary_chain = _RecordingRunnable([grounded])
    category_chain = _RecordingRunnable(
        [
            MerchantEnrichment(
                canonical_name="Westside",
                category="shopping",
                is_essential=False,
                risk_flag=None,
                lifestyle_dim="aspirational",
                recurring_type="adhoc",
            )
        ]
    )
    fallback._summary_chain = summary_chain
    fallback._category_chain = category_chain

    result = fallback.enrich("Westside")

    assert result is not None
    assert result.category == "shopping"
    assert result.lifestyle_dim == "aspirational"
    assert fallback._search.queries == ["Westside"]

    # The summarize step must have been grounded in the web content, not a
    # bare-name guess.
    assert len(summary_chain.calls) == 1
    assert "web results" in summary_chain.calls[0]["name"].lower()

    used_summary = category_chain.calls[0]["summary"].lower()
    assert "groceries" not in used_summary
    assert "fashion" in used_summary


def test_returns_none_when_web_search_finds_nothing_usable():
    """Web search runs but the grounded summary still comes back as
    'No information found' — genuinely unresolved, not a guess."""
    fallback = _LLMFallback()
    fallback._failed = False
    fallback._search = _FakeWebSearch("Irrelevant, unrelated web noise.")

    summary_chain = _RecordingRunnable([{"summary": "No information found"}])
    category_chain = _RecordingRunnable([])  # must never be reached
    fallback._summary_chain = summary_chain
    fallback._category_chain = category_chain

    result = fallback.enrich("Totally Obscure Merchant")

    assert result is None
    assert fallback._search.queries == ["Totally Obscure Merchant"]
    assert len(category_chain.calls) == 0


def test_returns_none_when_web_search_disabled():
    """Without web search enabled (the default, no API key configured), the
    fallback resolves nothing at all — no guess, no chain calls."""
    fallback = _LLMFallback()
    fallback._failed = False
    fallback._search = _FakeWebSearch(None, enabled=False)

    summary_chain = _RecordingRunnable([])  # must never be reached
    category_chain = _RecordingRunnable([])  # must never be reached
    fallback._summary_chain = summary_chain
    fallback._category_chain = category_chain

    result = fallback.enrich("Westside")

    assert result is None
    assert fallback._search.queries == []  # never called when disabled
    assert len(summary_chain.calls) == 0
    assert len(category_chain.calls) == 0


def test_returns_none_when_web_search_returns_nothing():
    """Web search is enabled but comes back empty (no results, or the API
    call failed) — resolves to None, not a guess."""
    fallback = _LLMFallback()
    fallback._failed = False
    fallback._search = _FakeWebSearch(None, enabled=True)

    summary_chain = _RecordingRunnable([])  # must never be reached
    category_chain = _RecordingRunnable([])  # must never be reached
    fallback._summary_chain = summary_chain
    fallback._category_chain = category_chain

    result = fallback.enrich("Westside")

    assert result is None
    assert fallback._search.queries == ["Westside"]
    assert len(summary_chain.calls) == 0
