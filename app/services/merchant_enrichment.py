"""Merchant enrichment: cache -> dictionary -> web-search-grounded LangChain
fallback -> cache.

ISSUE #7 (module + cache + dictionary) and ISSUE #7b (LangChain fallback).

PRIVACY BOUNDARY (docs/enrichment_api.md):
    The ONLY thing that ever leaves this process is a merchant NAME string
    (plus a fixed disambiguation suffix for the web search call). No
    narrations, no amounts, no dates, no person payees. Asserted by the
    payload test in Issue #9.

DESIGN NOTES
------------
Lookup order is deliberate — each tier is ~1000x more expensive than the last:
    1. SQLite cache   ~0.1ms   every unresolved merchant costs ONE fallback
                                call EVER
    2. dictionary     ~0.01ms  ~70 merchants, offline, free
    3. web search + LangChain classify   ~1-3s   LangSearch grounds it,
       then a local model (via LangChain) turns the real content into a
       structured MerchantEnrichment. There is deliberately NO "guess the
       category from the bare name alone" step — a local model asked to
       describe a merchant it doesn't actually know will confidently invent
       a plausible-sounding answer instead of admitting ignorance (observed
       live: "Westside" -> hallucinated "groceries"). Without web search
       enabled and a working LANGSEARCH_API_KEY, tier 3 does nothing at all
       and the merchant falls through to unknown() rather than a guess.

Aliases exist because real UPI narrations TRUNCATE names: the DP model returns
"SWIGGY LI", "FLIPKAR T", "JIO RECHA", "IRCTCT OUR". Exact-match on the
canonical name would miss all of them.

Every failure path returns MerchantEnrichment.unknown() — this module NEVER
raises and NEVER blocks the pipeline. A missing merchant costs us L3 accuracy;
a crash costs us the whole report.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from pathlib import Path

from app.core.config import settings
from app.schemas.statements import MerchantEnrichment

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parents[2]
_DICT_PATH = _BASE_DIR / "data" / "india_merchants.json"
_CACHE_PATH = _BASE_DIR / "data" / "enrichment_cache.db"

# docs/taxonomy.md (Issue #4, FROZEN): is_essential, lifestyle_dim, and
# risk_flag are a DETERMINISTIC function of category — every entry in
# data/india_merchants.json bakes this in consistently, and the LLM must
# never be trusted to guess these three independently. An LLM asked for all
# five fields at once can (and does, observed live on "Westside") return a
# category and a lifestyle_dim that don't belong together — internally
# valid values, individually, that are still a wrong taxonomy pairing.
# _coerce() derives is_essential/lifestyle_dim/risk_flag from category here;
# only category and recurring_type are actually trusted from the model.
_CATEGORY_TAXONOMY: dict[str, tuple[bool, str, str | None]] = {
    # category: (is_essential, lifestyle_dim, risk_flag)
    "groceries": (True, "essential", None),
    "utilities": (True, "essential", None),
    "telecom": (True, "essential", None),
    "fuel": (True, "essential", None),
    "transport": (True, "essential", None),
    "education": (True, "essential", None),
    "healthcare": (True, "essential", None),
    "food_delivery": (False, "aspirational", None),
    "quick_commerce": (False, "aspirational", None),
    "shopping": (False, "aspirational", None),
    "entertainment": (False, "aspirational", None),
    "travel": (False, "aspirational", None),
    "personal_care": (False, "aspirational", None),
    "dining": (False, "aspirational", None),
    "investments": (False, "commitment", None),
    "insurance": (True, "commitment", None),
    "rent": (True, "commitment", None),
    "loan_emi": (True, "commitment", None),
    "bnpl_lending": (False, "leverage", "bnpl_lending"),
    "gambling": (False, "risk", "gambling"),
    "crypto": (False, "risk", "crypto"),
    "p2p_transfer": (False, "neutral", None),
    "cash_withdrawal": (False, "neutral", None),
    "gig_platform": (False, "neutral", None),
    "bank_charges": (False, "neutral", None),
    "other": (False, "neutral", None),
}
_VALID_CATEGORIES = set(_CATEGORY_TAXONOMY)
_VALID_RECUR = {"adhoc", "subscription", "emi_like", "payout_source"}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_SUFFIX_RE = re.compile(
    r"\b(pvt|private|ltd|limited|llp|inc|corp|corporation|co|company|"
    r"technologies|technology|solutions|services|india|enterprises)\b",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """'SWIGGY LI  ' -> 'swiggy li' ; 'Bundl Technologies Pvt Ltd' -> 'bundl'.

    Lowercase, strip punctuation and corporate suffixes, collapse whitespace.
    Dictionary aliases are stored pre-normalized so lookup is a dict hit.
    """
    s = (name or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SUFFIX_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Dictionary (tier 2)
# ---------------------------------------------------------------------------


class _Dictionary:
    """Alias -> MerchantEnrichment, loaded once."""

    def __init__(self) -> None:
        self._by_alias: dict[str, MerchantEnrichment] = {}
        self._gig_aliases: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not _DICT_PATH.exists():
            logger.warning("Merchant dictionary not found: %s", _DICT_PATH)
            return
        try:
            raw = json.loads(_DICT_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Merchant dictionary unreadable: %s", exc)
            return

        for entry in raw.get("merchants", []):
            try:
                enrichment = MerchantEnrichment(
                    canonical_name=entry["canonical_name"],
                    category=entry["category"],
                    is_essential=entry["is_essential"],
                    risk_flag=entry.get("risk_flag"),
                    lifestyle_dim=entry["lifestyle_dim"],
                    recurring_type=entry["recurring_type"],
                )
            except Exception as exc:
                logger.warning("Bad dictionary entry %s: %s", entry.get("canonical_name"), exc)
                continue
            for alias in entry.get("aliases", []):
                self._by_alias[normalize_name(alias)] = enrichment

        gig = raw.get("gig_payout_sources", {})
        self._gig_aliases = {normalize_name(a) for a in gig.get("aliases", [])}
        logger.info("Merchant dictionary: %d aliases loaded", len(self._by_alias))

    def lookup(self, normalized: str) -> MerchantEnrichment | None:
        if not normalized:
            return None

        # exact
        hit = self._by_alias.get(normalized)
        if hit is not None:
            return hit

        # prefix — handles truncation: "swiggy li" matches alias "swiggy li",
        # but "flipkar" (cut even shorter) should still find "flipkart".
        # Only match aliases >= 4 chars to avoid "jio" matching everything.
        for alias, enrichment in self._by_alias.items():
            if (
                len(alias) >= 4
                and len(normalized) >= 4
                and (normalized.startswith(alias) or alias.startswith(normalized))
            ):
                return enrichment
        return None

    def is_gig_source(self, normalized: str) -> bool:
        return any(
            normalized.startswith(a) or a.startswith(normalized)
            for a in self._gig_aliases
            if len(a) >= 4 and len(normalized) >= 4
        )


# ---------------------------------------------------------------------------
# Cache (tier 1)
# ---------------------------------------------------------------------------


class _Cache:
    """SQLite. Makes every unknown merchant cost exactly one LLM call, ever."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS enrichment ("
                "  normalized_name TEXT PRIMARY KEY,"
                "  payload TEXT NOT NULL,"
                "  source TEXT NOT NULL,"
                "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ")"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=5.0)

    def get(self, normalized: str) -> MerchantEnrichment | None:
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT payload FROM enrichment WHERE normalized_name = ?",
                    (normalized,),
                ).fetchone()
            if row:
                return MerchantEnrichment(**json.loads(row[0]))
        except Exception as exc:
            logger.debug("Cache read failed for %r: %s", normalized, exc)
        return None

    def put(self, normalized: str, enrichment: MerchantEnrichment, source: str) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO enrichment (normalized_name, payload, source)"
                    " VALUES (?, ?, ?)",
                    (normalized, enrichment.model_dump_json(), source),
                )
        except Exception as exc:
            logger.debug("Cache write failed for %r: %s", normalized, exc)


# ---------------------------------------------------------------------------
# Tier 3 — ISSUE #7b (LangChain): web search -> summarize -> categorize.
# No local-only guessing step.
# ---------------------------------------------------------------------------

_NO_INFO = "No information found"

_SUMMARY_PROMPT = """You are given an Indian merchant name from a UPI bank transaction.
The name is often TRUNCATED or abbreviated ("SWIGGY LI" = Swiggy, "FLIPKAR T" = Flipkart,
"JIO RECHA" = Jio Recharge).

Describe the business in 2-3 lines: what it sells, its industry, its domain.

Rules:
- Use ONLY your existing knowledge. Do NOT invent details.
- If you do not recognise this merchant, return EXACTLY: "No information found"
- Do not guess from the letters. "OGI12" is not "OGI International Limited"
  unless you actually know that company.

Merchant name: {name}

Return JSON: {{"summary": "<2-3 lines, or 'No information found'>"}}"""


_CATEGORIZE_PROMPT = """Classify an Indian merchant for a credit-assessment system.

You are given the merchant name AND a description of its business. Use the
description — it is more reliable than the truncated name.

`category` — pick EXACTLY one:
  groceries, utilities, telecom, fuel, transport, education, healthcare,
  food_delivery, quick_commerce, shopping, entertainment, travel,
  personal_care, dining, investments, insurance, rent, loan_emi,
  bnpl_lending, gambling, crypto, p2p_transfer, cash_withdrawal,
  gig_platform, bank_charges, other

`risk_flag`: gambling | bnpl_lending | crypto | null
`lifestyle_dim`: essential | aspirational | commitment | leverage | risk | neutral
`recurring_type`: adhoc | subscription | emi_like | payout_source

Rules:
- is_essential=true ONLY for: groceries, utilities, telecom, fuel, transport,
  education, healthcare, insurance, rent, loan_emi
- lifestyle_dim must match the category's group (see the taxonomy).
- If the description says "No information found" or is empty, return
  category "other", lifestyle_dim "neutral", recurring_type "adhoc".
  Do NOT guess.
- canonical_name is the clean brand name, e.g. "Swiggy".

Merchant name: {name}
What we know about it: {summary}"""


class _WebSearch:
    """Merchant name -> web results -> meta descriptions.

    Adapted from the reference pipeline's LangSearch node. This is the ONLY
    source of "what is this merchant" — there is no local-only guessing step
    before it (see `_LLMFallback.enrich`). Fires for every dictionary/cache
    miss when enabled; each merchant is cached forever after its first
    resolution, so the per-statement cost stays bounded to genuinely novel
    merchants.

    PRIVACY: the outbound query is the merchant name plus a fixed, static
    disambiguation suffix ("India merchant brand") — never anything derived
    from the user's transaction (no narration/amount/date/balance/person
    payees). The suffix exists because a bare name like "Westside" collides
    with unrelated results (NYC's West Side, a dictionary entry for the
    adjective) and starves tier 3b of anything useful about the actual
    Indian retail brand.
    """

    def __init__(self) -> None:
        self._enabled = bool(
            getattr(settings, "ENRICHMENT_WEBSEARCH_ENABLED", False)
            and getattr(settings, "LANGSEARCH_API_KEY", "")
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def search(self, name: str) -> str | None:
        """Return concatenated descriptions, or None."""
        if not self._enabled:
            return None
        try:
            import httpx

            timeout = getattr(settings, "ENRICHMENT_WEBSEARCH_TIMEOUT", 8.0)
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    "https://api.langsearch.com/v1/web-search",
                    headers={
                        "Authorization": f"Bearer {settings.LANGSEARCH_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    # Name + fixed disambiguation suffix — see class docstring.
                    json={
                        "query": f"{name} India merchant brand",
                        "count": 3,
                        "summary": True,
                        "freshness": "noLimit",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            # LangSearch nests results under "data", not "SearchData" — the
            # old key name silently returned zero results on every real call.
            results = (data.get("data", {}).get("webPages", {}).get("value", []))[:3]
            if not results:
                return None

            blocks: list[str] = []
            for item in results:
                text = item.get("summary") or item.get("snippet") or ""
                if text:
                    blocks.append(text[:500])  # cap — we only need the gist
            return "\n\n".join(blocks) if blocks else None

        except Exception as exc:
            logger.info("Web search failed for %r: %s", name, exc)
            return None


class _LLMFallback:
    """Web-search-grounded classification. No blind local guessing.

    Chain:  web search   (LangSearch — the ONLY source of "what is this
                          merchant"; must return real content or the
                          merchant stays unresolved)
         -> summarize    (LangChain + local Ollama, grounded in the web
                          results — never in the bare name alone)
         -> categorize   (LangChain + local Ollama, structured output)

    There used to be a step that asked the local model to describe a
    merchant from its name alone before ever touching the web. Removed: a
    1.5B model asked "what is Westside?" with no grounding will confidently
    answer "groceries" rather than admit it doesn't know — a fabricated
    category is worse than an unresolved one. So this tier now does nothing
    at all unless ENRICHMENT_WEBSEARCH_ENABLED and a working
    LANGSEARCH_API_KEY are both set; without them, every merchant that
    misses the dictionary falls straight through to
    MerchantEnrichment.unknown() instead of a guess.

    Two LLM steps (summarize, then categorize) instead of name->category
    directly: the reference pipeline's insight. Asking the model to reason
    over an actual description gives it something concrete to work from,
    rather than free-associating from a brand name it may not know at all.

    Lazily constructed so the engine starts fine without langchain installed.
    """

    def __init__(self) -> None:
        self._summary_chain = None
        self._category_chain = None
        self._failed = False
        self._search = _WebSearch()

    def _build(self) -> bool:
        if self._failed:
            return False
        if self._summary_chain is not None:
            return True
        try:
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_ollama import ChatOllama

            model = getattr(settings, "ENRICHMENT_LLM_MODEL", "qwen2.5:1.5b-instruct")
            base = ChatOllama(
                model=model,
                base_url=settings.OLLAMA_HOST,
                temperature=0,
                request_timeout=25,
            )

            # Step 1: free-text summary, JSON-wrapped.
            self._summary_chain = (
                ChatPromptTemplate.from_template(_SUMMARY_PROMPT)
                | base.bind(format="json")
                | JsonOutputParser()
            )

            # Step 2: structured classification. with_structured_output binds
            # the Pydantic model, so the SHAPE is guaranteed. The VALUES still
            # need checking — see _coerce().
            self._category_chain = ChatPromptTemplate.from_template(
                _CATEGORIZE_PROMPT
            ) | base.with_structured_output(MerchantEnrichment)
            return True
        except Exception as exc:
            logger.warning("LLM fallback unavailable (%s) — neutral defaults", exc)
            self._failed = True
            return False

    def enrich(self, name: str) -> MerchantEnrichment | None:
        # Web search is the ONLY entry point — no blind "guess from the bare
        # name" step. Without it enabled (and a working API key), this tier
        # resolves nothing, so there's no point even building the LangChain
        # chains.
        if not self._search.enabled:
            return None
        if not self._build():
            return None

        web = self._search.search(name)
        if not web:
            return None

        # Summarize the scraped text with the local model, grounded in real
        # content. The web blocks are noisy; we want 2-3 clean lines for
        # the categorizer.
        try:
            out = self._summary_chain.invoke({"name": f"{name}\n\nWeb results:\n{web}"})
            summary = (out or {}).get("summary", "").strip() if isinstance(out, dict) else ""
        except Exception as exc:
            logger.info("Summary step failed for %r: %s", name, exc)
            summary = web[:400]  # fall back to raw text; better than nothing

        if not summary or _NO_INFO.lower() in summary.lower():
            # Web search found something, but nothing usable came out of it.
            # Genuinely unknown — cache it so we don't retry a hopeless name.
            return None

        # --- summary -> category ---
        try:
            result = self._category_chain.invoke({"name": name, "summary": summary})
        except Exception as exc:
            logger.info("Categorize step failed for %r: %s", name, exc)
            return None
        return _coerce(result, name)


def _coerce(result, name: str) -> MerchantEnrichment:
    """Validate LLM output against the taxonomy, and derive the fields that
    aren't actually the model's to decide.

    with_structured_output guarantees the SHAPE; it does not guarantee the
    VALUES are in our taxonomy, or that is_essential/lifestyle_dim/risk_flag
    agree with category. Only `category` and `recurring_type` are trusted
    from the LLM — the other three are looked up from _CATEGORY_TAXONOMY so
    they can never disagree with category (docs/taxonomy.md is the source
    of truth, not the model's guess).
    """
    try:
        if isinstance(result, dict):
            result = MerchantEnrichment(**result)
        if result.category not in _CATEGORY_TAXONOMY or result.recurring_type not in _VALID_RECUR:
            logger.info("LLM returned off-taxonomy values for %r — coercing", name)
            return MerchantEnrichment.unknown(name)
        is_essential, lifestyle_dim, risk_flag = _CATEGORY_TAXONOMY[result.category]
        result.is_essential = is_essential
        result.lifestyle_dim = lifestyle_dim
        result.risk_flag = risk_flag
        return result
    except Exception:
        return MerchantEnrichment.unknown(name)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MerchantEnrichmentService:
    def __init__(self) -> None:
        self._dictionary = _Dictionary()
        self._cache = _Cache(_CACHE_PATH)
        self._fallback = _LLMFallback()

    def enrich_one(self, name: str) -> MerchantEnrichment:
        normalized = normalize_name(name)
        if not normalized:
            return MerchantEnrichment.unknown(name)

        # 1. cache
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached

        # 2. dictionary
        hit = self._dictionary.lookup(normalized)
        if hit is not None:
            self._cache.put(normalized, hit, source="dictionary")
            return hit

        # 3. web-search-grounded fallback (Issue #7b) — no-op unless
        # ENRICHMENT_WEBSEARCH_ENABLED + LANGSEARCH_API_KEY are both set.
        if getattr(settings, "ENRICHMENT_LLM_ENABLED", True):
            fallback_hit = self._fallback.enrich(name)
            if fallback_hit is not None:
                self._cache.put(normalized, fallback_hit, source="llm")
                return fallback_hit

        # 4. give up safely — cached so we don't retry a hopeless name forever
        fallback = MerchantEnrichment.unknown(name)
        self._cache.put(normalized, fallback, source="unknown")
        return fallback

    def enrich(self, names: list[str]) -> list[MerchantEnrichment]:
        """Batch, order-preserving. Deduped so 6 IRCTC rows = 1 lookup."""
        unique = {normalize_name(n): n for n in names}
        resolved = {norm: self.enrich_one(orig) for norm, orig in unique.items()}
        return [resolved[normalize_name(n)] for n in names]

    def is_gig_payout_source(self, name: str) -> bool:
        """True if a CREDIT from this merchant is gig income (Gig Hustler)."""
        return self._dictionary.is_gig_source(normalize_name(name))


_service: MerchantEnrichmentService | None = None


def get_enrichment_service() -> MerchantEnrichmentService:
    global _service
    if _service is None:
        _service = MerchantEnrichmentService()
    return _service
