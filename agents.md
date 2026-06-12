# CreditMitra Backend — Agent Instructions

> **Repository**: `CreditMitra-engine`
> **Stack**: Python 3.10+ · FastAPI · Pydantic v2 · Uvicorn · Docling · Ollama

---

## 1. Project Identity

CreditMitra is a **bank-statement analysis pipeline** that:

1. Accepts PDF bank statements via a REST API.
2. Extracts tabular transaction data using **Docling** (document AI).
3. Enriches each transaction with a predicted **payee name** via a locally-hosted **Ollama** LLM (`payee-lora:latest`).
4. Persists task state and results as **local JSON files** — there is **no database**.

The API is consumed by a React/TypeScript dashboard frontend (`CreditMitra-dashboard`). In the future, the frontend will be packaged as a desktop app via **Tauri**, and this backend may be embedded or sidecar'd alongside it.

---

## 2. Architecture Overview

```
backend/
├── app/
│   ├── main.py              # FastAPI app factory, CORS, router mounting
│   ├── api/
│   │   └── v1/
│   │       └── statements.py # Route handlers (upload, status polling)
│   ├── core/
│   │   └── config.py         # Pydantic Settings (env/.env, Ollama config, paths)
│   ├── schemas/
│   │   └── statements.py     # Pydantic v2 request/response models
│   └── services/
│       ├── extraction.py     # Docling PDF parsing + Ollama payee prediction
│       └── task_store.py     # JSON file-based task persistence
├── data/                     # Runtime data directory (auto-created)
│   ├── tasks/                # One JSON file per task_id
│   └── uploads/              # Temporary uploaded PDFs (deleted after processing)
├── requirements.txt
├── pyproject.toml            # Ruff + mypy configuration
└── Dockerfile
```

### Layered Responsibilities

| Layer        | Directory          | Responsibility                                                      |
| ------------ | ------------------ | ------------------------------------------------------------------- |
| **Routing**  | `app/api/v1/`      | HTTP concerns only — validation, status codes, background dispatch  |
| **Schemas**  | `app/schemas/`     | Pydantic models for request/response serialization                  |
| **Services** | `app/services/`    | All business logic — extraction, prediction, task persistence       |
| **Core**     | `app/core/`        | Cross-cutting concerns — settings, future middleware, shared utils  |

---

## 3. Critical Constraints — Read Before Writing Any Code

### 3.1 No Database

All state is stored as flat JSON files under `data/tasks/`. **Do not** introduce SQLAlchemy, SQLModel, SQLite, PostgreSQL, or any ORM. If you need to persist new data, use the existing `task_store.py` pattern — a new JSON file keyed by a UUID.

### 3.2 No Authentication / Authorization

There are **no auth endpoints, no JWT, no OAuth, no API keys**. This is intentional — the app is designed to eventually run as a local desktop tool. Do not add authentication middleware or guards unless the user explicitly requests it.

### 3.3 No Rate Limiting

No rate limiting or throttling. Same reasoning — eventual desktop app. Do not add `slowapi` or similar.

### 3.4 Background Processing

PDF processing (Docling extraction + Ollama prediction) is a **long-running operation** and must **always** run in a FastAPI `BackgroundTasks` handler. The API returns immediately with a `task_id`, and the frontend polls `GET /status/{task_id}` until completion. Never make the upload endpoint synchronous.

### 3.5 Temporary Files Cleanup

Uploaded PDFs are saved to `data/uploads/` and **must be deleted** in the `finally` block of `process_pdf_task()` after processing completes or fails.

---

## 4. API Surface

All routes are prefixed with `/api/v1`.

| Method | Path                              | Purpose                                      |
| ------ | --------------------------------- | -------------------------------------------- |
| `POST` | `/api/v1/statements/process`      | Upload a PDF, returns `{ task_id, status }`   |
| `GET`  | `/api/v1/statements/status/{id}`  | Poll task status and retrieve results         |
| `GET`  | `/health`                         | Simple health check (no prefix)               |

### Task Lifecycle

```
pending → processing → completed | failed
```

- `pending` — task created, file saved, background job dispatched.
- `processing` — Docling extraction has begun.
- `completed` — results written to task JSON, includes `transactions[]`.
- `failed` — error string written to task JSON.

---

## 5. External Service Dependencies

### 5.1 Ollama (Local LLM)

- **Default host**: `http://127.0.0.1:11434` (override via `OLLAMA_HOST` env var)
- **Default model**: `payee-lora:latest` (override via `OLLAMA_MODEL` env var)
- The backend calls `POST /api/generate` with `stream: false` and `raw: true`.
- If Ollama is unreachable, payee prediction silently returns an empty string — it does **not** crash the task.

### 5.2 Docling

- Used via `docling.document_converter.DocumentConverter` for PDF table extraction.
- Heavy dependency — first invocation may download models. Be aware of cold-start latency.

---

## 6. Configuration

All config is centralized in `app/core/config.py` using `pydantic-settings`. Values can be set via:

1. Environment variables (highest priority).
2. A `.env` file in the `backend/` root.
3. Defaults hardcoded in the `Settings` class.

| Variable       | Default                          | Description                        |
| -------------- | -------------------------------- | ---------------------------------- |
| `OLLAMA_HOST`  | `http://127.0.0.1:11434`        | Ollama server URL                  |
| `OLLAMA_MODEL` | `payee-lora:latest`              | Model name for payee prediction    |
| `DATA_DIR`     | `./data`                         | Root for tasks/ and uploads/       |

---

## 7. Coding Conventions

### Style & Linting

- **Formatter/Linter**: Ruff (configured in `pyproject.toml` — line length 100, rules `E`, `F`, `I`).
- **Type checker**: mypy in strict mode with `ignore_missing_imports = true`.
- Run checks: `ruff check .` and `mypy app/`.

### Naming

- Files: `snake_case.py`
- Classes: `PascalCase` (Pydantic models, settings)
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

### Pydantic

- Use **Pydantic v2** syntax exclusively. Do not use `class Config` — use `model_config = SettingsConfigDict(...)`.
- All API request/response bodies must have a corresponding schema in `app/schemas/`.

### Logging

- Use `logging.getLogger(__name__)` — do not use `print()` for any operational output.
- Log errors at `logger.error()`, info at `logger.info()`.

### Error Handling

- Use `HTTPException` for API errors with appropriate status codes.
- Background tasks must catch all exceptions and write `"failed"` + error message to the task store. They must **never** crash silently.

---

## 8. Adding New Features — Step-by-Step

When adding a new capability (e.g., CSV export, analytics endpoint):

1. **Schema first** — Define Pydantic models in `app/schemas/` (or create a new file).
2. **Service logic** — Write business logic in `app/services/` (new file if it's a distinct domain).
3. **Route handler** — Add the endpoint in `app/api/v1/` (new file + register router in `main.py`).
4. **Config** — If the feature needs new settings, add them to `app/core/config.py`.
5. **Never** put business logic directly in route handlers — keep them thin.

### Adding a New Router

```python
# app/api/v1/analytics.py
from fastapi import APIRouter
router = APIRouter()

@router.get("/summary")
async def get_summary():
    ...
```

```python
# app/main.py — register it
from app.api.v1 import analytics
app.include_router(analytics.router, prefix=f"{settings.API_V1_STR}/analytics", tags=["analytics"])
```

---

## 9. CORS Policy

Allowed origins are explicitly listed in `app/main.py`:

```python
allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "tauri://localhost"]
```

- `localhost:5173` — Vite dev server for the React frontend.
- `tauri://localhost` — Future Tauri desktop app origin.

When adding new origins (e.g., Electron), update this list. **Do not** use `allow_origins=["*"]` in any environment.

---

## 10. Running the Backend

```bash
cd backend
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

The server starts at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

### Docker

```bash
docker build -t creditmitra-engine .
docker run -p 8000:8000 creditmitra-engine
```

> **Note**: Ollama must be reachable from inside the container. Use `--network host` or set `OLLAMA_HOST` to the host machine's IP.

---

## 11. Testing

- Test framework: **pytest**.
- Run tests: `pytest` from the `backend/` directory.
- Write tests in a `tests/` directory mirroring the `app/` structure.
- For API endpoint tests, use FastAPI's `TestClient`:

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
```

---

## 12. Future Considerations

- **Tauri sidecar**: This backend may be bundled as a sidecar process in a Tauri app. Keep startup fast, avoid global state that assumes a long-lived server.
- **Multiple file formats**: Docling supports more than PDF. The architecture should accommodate CSV/Excel ingestion in the future — add new extractors in `app/services/` without modifying the existing PDF flow.
- **Export endpoints**: CSV/JSON export of processed transactions is a likely next feature. Design it as a new router under `app/api/v1/export.py`.
