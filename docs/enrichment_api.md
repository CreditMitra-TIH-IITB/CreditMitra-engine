# Merchant Enrichment API Contract — Issue #2

**Status:** contract only. Today, `app/services/merchant_enrichment.py` (Issue
#7/#7b) resolves merchants in-process (cache → `data/india_merchants.json` →
LangChain/Ollama). This document specifies the HTTP contract for the future
"merchant-server" — a standalone service the engine can call instead of (or
alongside) the in-process resolver, so a device without a local Ollama model
can still get enrichment. Response shape is `MerchantEnrichment` (see
`app/schemas/statements.py`), same taxonomy as `docs/taxonomy.md` (#4).

---

## Request

```
POST /api/v1/merchants/enrich
Header: X-API-Key: <key>
Content-Type: application/json
```

```json
{ "merchants": ["Swiggy", "Jio", "IRCTC"] }
```

- `merchants` — a flat list of merchant NAME strings, in the order enrichment
  is wanted. Names may be truncated/abbreviated (as UPI narrations produce
  them, e.g. `"SWIGGY LI"`, `"FLIPKAR T"`) — the server applies the same
  normalization/alias matching as the in-process resolver.

## Response — `200 OK`

```json
{
  "results": [
    {
      "canonical_name": "Swiggy",
      "category": "food_delivery",
      "is_essential": false,
      "risk_flag": null,
      "lifestyle_dim": "aspirational",
      "recurring_type": "adhoc"
    },
    {
      "canonical_name": "Jio",
      "category": "telecom",
      "is_essential": true,
      "risk_flag": null,
      "lifestyle_dim": "essential",
      "recurring_type": "subscription"
    },
    {
      "canonical_name": "IRCTC",
      "category": "transport",
      "is_essential": true,
      "risk_flag": null,
      "lifestyle_dim": "essential",
      "recurring_type": "adhoc"
    }
  ]
}
```

- `results` is **always the same length, in the same order** as the request's
  `merchants` array — the caller zips request↔response by index. Never
  reordered, never omitted, never partial.
- Every `category` / `lifestyle_dim` / `recurring_type` / `risk_flag` value is
  one from `docs/taxonomy.md`. A caller must never see an off-taxonomy value.

### Unknown merchants

A name the server can't resolve (dictionary miss, LLM timeout, off-taxonomy
LLM output) still returns a record — never an error, never a dropped index:

```json
{
  "canonical_name": "<the original input string>",
  "category": "other",
  "is_essential": false,
  "risk_flag": null,
  "lifestyle_dim": "neutral",
  "recurring_type": "adhoc"
}
```

This mirrors `MerchantEnrichment.unknown()` in
`app/schemas/statements.py` — the same safe default the in-process resolver
falls back to. The scorer must never treat "unresolved" as a signal by itself.

## Errors

| Status | Cause |
|---|---|
| `401` | missing or invalid `X-API-Key` |
| `422` | malformed body — not JSON, `merchants` missing, not a list, or contains a non-string element |

A `4xx`/`5xx` response applies to the **whole batch**, not individual names —
per-name failures are absorbed into the `"other"/"neutral"/"adhoc"` fallback
above, not surfaced as partial errors.

## Privacy guarantee

The request body contains **merchant name strings and nothing else.** No
transaction narrations, amounts, dates, balances, or person-classified
payees ever leave the device/process boundary. `payee_type == "person"` rows
must be filtered out by the caller before building the `merchants` list —
the server has no way to enforce this itself, so it's a caller obligation.

Enforced on the in-process path by the payload test in Issue #9 (asserts the
outbound LLM/API call site only ever receives merchant names, never full
transaction rows).
