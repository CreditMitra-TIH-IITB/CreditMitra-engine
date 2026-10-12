# CreditMitra Engine — Agent Instructions

> **Repository**: `CreditMitra-engine`
> **Stack**: Python ≥ 3.10 · FastAPI · Pydantic v2 · Docling + pdfplumber · Ollama · onnxruntime

Read this before changing code. `README.md` covers the product, `docs/RUNNING.md`
local setup, `docs/DEPLOY.md` Docker, `docs/taxonomy.md` the merchant taxonomy and
scoring weights.

---

## 1. What the engine does

A bank-statement PDF goes in; transactions, a payee per row, person/merchant
labels, merchant categories, and a 300–900 lifestyle credit report come out.
One background task per statement (`process_pdf_task`):

1. **Extract** — Docling tables, pdfplumber fallback (`extraction.py`,
   `statement_layout.py`, `parsing.py`). Columns are matched by keyword; a
   boxed-header/borderless-body layout is rebuilt from word positions. An
   unreadable statement fails the task — it never completes with 0 rows.
2. **Payee** — local Ollama model (`payee-lora:latest`), then `ground_payee`
   rejects names that are not in the narration.
3. **Classify** — ONNX embedding + AttentionMLP, person-name guard on top
   (`merchant_classifier.py`); each distinct name once per statement.
4. **Enrich** — cache → dictionary → vector index → merchant server →
   optional web search (`merchant_enrichment.py`, `merchant_vectors.py`,
   `merchant_enrichment_client.py`).
5. **Score** — `feature_engineering.py` → `lifestyle_profile.py` →
   `archetype.py` → `credit_scorer.py`; weights only in `core/scoring_config.py`.

The React dashboard (`CreditMitra-dashboard`) is the client. The shared
merchant knowledge lives in `CreditMitra-merchant-server`.

---

## 2. Layout

```
app/
├── main.py                     # app, CORS, logging, lifespan (task recovery + warm-up)
├── api/v1/statements.py        # /process, /status/{id}, /report/{id}
├── core/config.py              # all settings (env / .env)
├── core/scoring_config.py      # every score weight, band and threshold
├── schemas/statements.py       # Pydantic contracts — additive changes only
└── services/                   # all business logic (see §1)
data/
├── india_merchants.json        # curated dictionary (tracked)
├── person_name_tokens.txt      # person-name guard word list (tracked)
├── models/                     # ONNX classifier files (not in git, ~1.2–2.4 GB)
├── tasks/  uploads/            # runtime (not in git)
└── enrichment_cache.db         # SQLite cache (not in git, regenerable)
experiments/incremental_lift/   # WP4 measurement harness (synthetic, Berka)
scripts/                        # dev start, model builds, e2e check, sample statements
deploy/                         # docker-compose for the whole stack
```

---

## 3. Rules that are easy to break

### Privacy
- **Only merchant names** (plus an opaque statement fingerprint) may leave the
  process, and only to the merchant server or the optional web-search tier.
  Never narrations, amounts, dates, balances, or person-classified payees.
  `tests/test_enrichment_privacy.py` pins the outbound payload.
- Uploaded PDFs are deleted in `process_pdf_task`'s `finally`; tasks left
  pending/processing by a crash are failed at startup and their uploads removed.
- Healthcare is excluded from every ratio and index (fair lending) —
  `test_healthcare_spend_never_changes_the_score` enforces it.

### Storage
- Task results are JSON files via `task_store.py`. Do not add an ORM or a
  database server. The SQLite enrichment cache is derived data and may be
  rebuilt at any time.

### Merchant labels
- `is_essential`, `lifestyle_dim` and `risk_flag` are derived from `category`
  (taxonomy), never chosen by a model.
- High-severity categories (gambling, bnpl_lending, crypto, loan_emi) must not
  be reachable by a prefix that merely starts a longer name, nor by the vector
  tier.

### Robustness
- Every stage degrades rather than fails (empty payee, heuristic classifier,
  `other` category, `report: null`). Only an unreadable statement fails a task.
- Long work runs in `BackgroundTasks`; the upload endpoint returns at once.

### Auth
- The engine has no authentication by design (local tool). Do not add it unless
  asked — the hosted-vs-desktop decision is open.

---

## 4. API

| Method | Path | Returns |
|---|---|---|
| `POST` | `/api/v1/statements/process` | `{task_id, status}` (PDF only, else 400) |
| `GET` | `/api/v1/statements/status/{id}` | status + transactions (404 if unknown) |
| `GET` | `/api/v1/statements/report/{id}` | status + `CreditRiskReport` or `null` |
| `GET` | `/health` | `{"status": "ok"}` |

Task lifecycle: `pending → processing → completed | failed`.

---

## 5. Configuration (`app/core/config.py`)

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `http://127.0.0.1:11434` / `payee-lora:latest` | Payee model |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keep the payee model resident |
| `WARMUP_ON_STARTUP` | `true` | Load models in the background at start |
| `ENRICHMENT_VECTOR_ENABLED` / `_THRESHOLD` | `true` / `0.75` | On-device fuzzy merchant tier |
| `ENRICHMENT_WEBSEARCH_ENABLED` / `LANGSEARCH_API_KEY` | `false` / — | Web-search tier (sends merchant names out) |
| `MERCHANT_ENRICHMENT_URL` / `MERCHANT_API_KEY` | — | Merchant server tier |
| `CORS_ORIGINS` / `CORS_ORIGIN_REGEX` | Vite dev origins + `tauri://localhost` | Never a wildcard |
| `DATA_DIR` | `./data` | Tasks, uploads, fingerprint secret |
| `LOG_LEVEL` | `INFO` | `app.*` loggers |

---

## 6. Conventions and checks

- Ruff (line length 100) and mypy strict on `app/` — CI runs both, plus pytest
  and a Docker build/boot check.
- `logging.getLogger(__name__)`, never `print`. Pydantic v2 only.
- New behaviour gets a test. Tests run without Ollama, models or network;
  `tests/conftest.py` disables warm-up and points `DATA_DIR` at a temp dir.

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy app/ && pytest -q
```

Local stack: `scripts/dev.ps1` (see `docs/RUNNING.md`). End-to-end check:
`scripts/e2e_check.py`. Containers: `deploy/docker-compose.yml`.
