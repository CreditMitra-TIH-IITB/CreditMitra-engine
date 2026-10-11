# Running CreditMitra locally

Three processes: **Ollama** (payee model), the **engine** (this repo, port 8000),
and the **dashboard** (`CreditMitra-dashboard`, port 5173). Nothing leaves the
machine unless you turn on web search or point the engine at a merchant server.

## One-time setup

Check out the engine and dashboard side by side:

```
CreditMitra-TIH-IITB/
├── CreditMitra-engine/
└── CreditMitra-dashboard/
```

**1. Engine environment** (Python ≥ 3.10)

```powershell
cd CreditMitra-engine
python -m venv venv
venv\Scripts\pip install -e ".[dev]"
```

**2. Classifier model files** — not in git (2.3 GB). Copy them from the
`CreditMitra-Merchant-Classification` repo's `models/` folder into
`data/models/` as listed in [`data/models/README.md`](../data/models/README.md).
`scaler.joblib` is easy to miss; without it the classifier silently falls back
to keyword heuristics.

**3. Payee model in Ollama** — install [Ollama](https://ollama.com), then build
the model from the merged checkpoint (`payee-merged`, from the finetune
workflow) and a llama.cpp checkout:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_payee_model.ps1 `
    -MergedDir ..\outputs\payee-merged -LlamaCpp ..\outputs\llama.cpp
```

This creates `payee-lora:latest` (Q8_0, ~1.6 GB). `ollama pull` will not work —
the model is not on a public registry.

**4. Dashboard**

```powershell
cd ..\CreditMitra-dashboard
npm install
```

## Start everything

```powershell
cd CreditMitra-engine
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
```

The script starts Ollama if needed, warns about anything missing, and opens the
engine and dashboard in their own windows. Then open http://localhost:5173.

Manually, or on macOS/Linux:

```bash
ollama serve                                   # if not already running
cd CreditMitra-engine && venv/bin/uvicorn app.main:app --port 8000
cd CreditMitra-dashboard && npm run dev
```

## What to expect

- **Startup**: the engine answers `/health` immediately; the payee model and
  the classifier keep loading in the background for 1–2 minutes. A statement
  uploaded before that just waits for them.
- **Processing time**: dominated by one payee-model call per row, ~0.2 s each,
  plus Docling's table extraction. A 150-row statement takes about a minute.
- **Uploaded PDFs are deleted** as soon as processing finishes. Results stay in
  `data/tasks/<task_id>.json`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Every payee is empty | Ollama not running, or no `payee-lora` model — engine log says `Ollama prediction failed` |
| People tagged as merchants, log says `Using dictionary + heuristic fallback` | Classifier files missing from `data/models/` |
| Task fails with `No transactions could be extracted… Table headers seen: …` | Statement layout not recognised — the headers in the message show what was found |
| "Credit report unavailable" on the dashboard | Rows parsed, but not enough dated/amount data to score — the engine log says why |
| Dashboard shows a network error | Engine not on port 8000, or dashboard origin not in `CORS_ORIGINS` |
