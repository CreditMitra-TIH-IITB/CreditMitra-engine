"""Issue #9 — privacy payload tests for merchant enrichment.

The guarantee (docs/enrichment_api.md, app/services/merchant_enrichment.py):
the ONLY thing that ever leaves this process during enrichment is a merchant
NAME string, a fixed static disambiguation suffix (web search), and — for
the categorize step — a summary the model generated about that same
merchant. Never anything derived from the user's account: no transaction
narrations, amounts, dates, balances, or person-classified payees.

There is no blind "guess from the bare name" step anymore — the LangChain
fallback only ever runs grounded in real web search content (see
_LLMFallback.enrich). These tests intercept the actual call sites
(chain.invoke / httpx.Client.post) so a future edit that widens a payload
silently (e.g. threading narration into a prompt "for better context")
fails loudly here.
"""

from __future__ import annotations

import httpx

from app.schemas.statements import MerchantEnrichment
from app.services.merchant_enrichment import _LLMFallback, _WebSearch


class _RecordingRunnable:
    """Fake LangChain runnable — records every payload dict it's invoked with."""

    def __init__(self, results):
        self.calls: list[dict] = []
        self._results = iter(results)

    def invoke(self, payload):
        self.calls.append(payload)
        return next(self._results)


class _RecordingHTTPClient:
    """Fake httpx.Client — records every POST payload, context-manager compatible."""

    def __init__(self, *args, **kwargs) -> None:
        self.posts: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        self.posts.append({"url": url, "headers": headers, "json": json})

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"SearchData": {"webPages": {"value": []}}}

        return _Resp()


class _DisabledSearch:
    """Forces the fallback off regardless of the local .env."""

    enabled = False

    def search(self, name):
        raise AssertionError("web search must not be consulted when disabled")


class _StubEnabledSearch:
    """A search tier that's on and returns a fixed, non-user-derived blob —
    stands in for a real LangSearch response so tests stay deterministic."""

    enabled = True

    def __init__(self, content: str = "Some public web content about the merchant."):
        self.queries: list[str] = []
        self._content = content

    def search(self, name):
        self.queries.append(name)
        return self._content


# ---------------------------------------------------------------------------
# LangChain fallback (web-search-grounded summarize + categorize)
# ---------------------------------------------------------------------------


def test_no_chains_touched_when_websearch_disabled():
    """With web search off (the default), enrich() must return immediately —
    no LangChain call, no chain even built."""
    fallback = _LLMFallback()
    fallback._search = _DisabledSearch()

    result = fallback.enrich("Totally Unknown Merchant XYZ")

    assert result is None
    assert fallback._summary_chain is None
    assert fallback._category_chain is None


def test_summary_chain_receives_name_plus_web_content_only():
    """Step 1 (grounded summary) must be invoked with exactly {'name': <str>},
    where the string is the merchant name plus ONLY the web search's own
    returned content — never narration/amount/date/balance."""
    fallback = _LLMFallback()
    fallback._failed = False
    fallback._search = _StubEnabledSearch("Some public web content about the merchant.")
    summary_chain = _RecordingRunnable([{"summary": "An Indian food delivery platform."}])
    category_chain = _RecordingRunnable(
        [
            MerchantEnrichment(
                canonical_name="Totally Unknown Merchant XYZ",
                category="other",
                is_essential=False,
                risk_flag=None,
                lifestyle_dim="neutral",
                recurring_type="adhoc",
            )
        ]
    )
    fallback._summary_chain = summary_chain
    fallback._category_chain = category_chain

    fallback.enrich("Totally Unknown Merchant XYZ")

    assert len(summary_chain.calls) == 1
    payload = summary_chain.calls[0]
    assert set(payload.keys()) == {"name"}
    assert payload["name"] == (
        "Totally Unknown Merchant XYZ\n\nWeb results:\nSome public web content about the merchant."
    )


def test_categorize_chain_receives_only_name_and_summary():
    """Step 2 (summary -> category) must be invoked with exactly
    {'name': <str>, 'summary': <str>} — no narration/amount/date/balance."""
    fallback = _LLMFallback()
    fallback._failed = False
    fallback._search = _StubEnabledSearch("Swiggy is an Indian food delivery platform.")
    summary_chain = _RecordingRunnable([{"summary": "An Indian food delivery platform."}])
    category_chain = _RecordingRunnable(
        [
            MerchantEnrichment(
                canonical_name="Swiggy",
                category="food_delivery",
                is_essential=False,
                risk_flag=None,
                lifestyle_dim="aspirational",
                recurring_type="adhoc",
            )
        ]
    )
    fallback._summary_chain = summary_chain
    fallback._category_chain = category_chain

    fallback.enrich("Swiggy")

    assert len(category_chain.calls) == 1
    payload = category_chain.calls[0]
    assert set(payload.keys()) == {"name", "summary"}
    assert payload["name"] == "Swiggy"
    assert isinstance(payload["summary"], str)


def test_llm_fallback_signature_takes_a_bare_name_string():
    """enrich() takes exactly one positional str — there is no parameter a
    caller could use to thread transaction context (narration/amount/date)
    through to the LLM, even by accident."""
    import inspect

    sig = inspect.signature(_LLMFallback.enrich)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert len(params) == 1
    assert params[0].name == "name"
    assert params[0].annotation in (str, "str")


# ---------------------------------------------------------------------------
# Tier 4 — web search (the only tier that leaves the device over the network)
# ---------------------------------------------------------------------------


def test_websearch_disabled_by_default():
    """The one tier that makes an external network call must default OFF.

    Checks the field's declared default, not an instantiated Settings() —
    a real .env (e.g. a dev machine that's opted into live web search
    testing) legitimately overrides the instance value without changing
    what "default" means here.
    """
    from app.core.config import Settings

    assert Settings.model_fields["ENRICHMENT_WEBSEARCH_ENABLED"].default is False


def test_websearch_payload_is_merchant_name_plus_static_suffix_only(monkeypatch):
    """When enabled, the outbound query is the merchant name plus a FIXED,
    static disambiguation suffix — never anything derived from the user's
    transaction (no narration/amount/date/balance)."""
    monkeypatch.setattr("app.core.config.settings.ENRICHMENT_WEBSEARCH_ENABLED", True)
    monkeypatch.setattr("app.core.config.settings.LANGSEARCH_API_KEY", "test-key")

    recorder = _RecordingHTTPClient()
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: recorder)

    ws = _WebSearch()
    ws.search("Totally Unknown Merchant XYZ")

    assert len(recorder.posts) == 1
    payload = recorder.posts[0]["json"]
    assert payload["query"].startswith("Totally Unknown Merchant XYZ")
    # Everything after the name must be a fixed string, not transaction data.
    suffix = payload["query"].removeprefix("Totally Unknown Merchant XYZ")
    assert "narration" not in suffix.lower()
    assert set(payload.keys()) == {"query", "count", "summary", "freshness"}


def test_websearch_noop_when_disabled(monkeypatch):
    """With the flag off (the default), search() must not touch the network
    at all — not even construct an httpx.Client."""
    monkeypatch.setattr("app.core.config.settings.ENRICHMENT_WEBSEARCH_ENABLED", False)

    def _fail_if_called(*a, **k):
        raise AssertionError("httpx.Client must not be constructed when web search is disabled")

    monkeypatch.setattr(httpx, "Client", _fail_if_called)

    ws = _WebSearch()
    result = ws.search("Some Merchant")
    assert result is None


# ---------------------------------------------------------------------------
# Wiring boundary — Issue #8's extraction.py step 4
# ---------------------------------------------------------------------------


def test_extraction_never_forwards_person_payees_or_transaction_context(monkeypatch):
    """End-to-end: given a mix of merchant- and person-classified payees with
    full transaction context attached (narration/amount/date/balance),
    process_pdf_task must forward ONLY the merchant payee name strings to
    the enrichment service — never a person payee, never a dict."""
    from app.services import extraction

    fake_rows = [
        {
            "date": "05-11-2025",
            "particulars": "UPI/DR/101895374870/SWIGGY LI/FOOD ORDER",
            "deposits": "",
            "withdrawals": "380.00",
            "balance": "42000.00",
            "txn_date": "2025-11-05",
            "amount": 380.0,
            "direction": "debit",
            "balance_val": 42000.0,
        },
        {
            "date": "06-11-2025",
            "particulars": "UPI/DR/101895374871/RAVI KUMAR/RENT",
            "deposits": "",
            "withdrawals": "15000.00",
            "balance": "27000.00",
            "txn_date": "2025-11-06",
            "amount": 15000.0,
            "direction": "debit",
            "balance_val": 27000.0,
        },
    ]
    monkeypatch.setattr(extraction, "extract_transactions", lambda pdf_path: fake_rows)

    payee_map = {
        "UPI/DR/101895374870/SWIGGY LI/FOOD ORDER": "Swiggy",
        "UPI/DR/101895374871/RAVI KUMAR/RENT": "Ravi Kumar",
    }
    monkeypatch.setattr(
        extraction, "predict_payee", lambda narration, client: payee_map.get(narration, "")
    )

    class _FakeClassifier:
        available = True

        def classify_batch(self, names):
            return [
                {"label": "merchant" if n == "Swiggy" else "person", "confidence": 0.9}
                for n in names
            ]

    monkeypatch.setattr(
        "app.services.merchant_classifier.MerchantClassifierService.get_instance",
        classmethod(lambda cls: _FakeClassifier()),
    )

    captured: dict = {}

    class _FakeEnrichmentService:
        def enrich(self, names):
            captured["names"] = list(names)
            for n in names:
                assert isinstance(n, str), "enrichment must receive name strings, never dicts"
            return [MerchantEnrichment.unknown(n) for n in names]

    monkeypatch.setattr(extraction, "get_enrichment_service", lambda: _FakeEnrichmentService())

    saved: dict = {}

    def fake_update_task_status(task_id, status, **kwargs):
        saved["status"] = status
        saved.update(kwargs)

    monkeypatch.setattr(extraction, "update_task_status", fake_update_task_status)

    extraction.process_pdf_task("test-task-id", "/nonexistent/fake.pdf")

    assert saved["status"] == "completed"
    assert captured["names"] == ["Swiggy"]
    assert "Ravi Kumar" not in captured["names"]
