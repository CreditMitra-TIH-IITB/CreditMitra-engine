# Deploying CreditMitra with Docker

`deploy/docker-compose.yml` runs the whole stack:

| Service | Image | Port | Notes |
|---|---|---|---|
| `engine` | this repo's `Dockerfile` | 8000 | Statement pipeline. CPU-only torch. |
| `merchant-server` | `CreditMitra-merchant-server` | internal | Shared merchant dictionary + curation loop |
| `dashboard` | `CreditMitra-dashboard` | 8080 | Static build behind nginx |
| `ollama` (optional) | `ollama/ollama` | 11434 | Only with `--profile ollama` |

## Prerequisites

- Docker with Compose v2. Give it **at least 6 GB of memory**: the engine holds
  the classifier's embedding model (1.2 GB as FP16, 2.4 GB as FP32) and
  Docling's layout models (~1.5 GB) while processing.
- The three repositories checked out side by side.
- Classifier models in `CreditMitra-engine/data/models/` (see
  `data/models/README.md`) — ideally with the FP16 build
  (`scripts/quantize_classifier.py build`). They are mounted read-only.
- A payee model in Ollama: either on the host (`scripts/build_payee_model.ps1`)
  or in the container (below).

## Run

```bash
cd CreditMitra-engine/deploy
cp .env.example .env          # set MERCHANT_API_KEY to a long random string
docker compose up --build -d
```

Then open http://localhost:8080. The engine API is at http://localhost:8000/docs.

The first statement after a fresh deploy is slow: Docling downloads its layout
models into the `engine-state` volume (once).

### Payee model in a container instead of on the host

```bash
# deploy/.env: OLLAMA_HOST=http://ollama:11434
# PAYEE_GGUF_DIR must point at the folder holding payee-q8_0.gguf
docker compose --profile ollama up -d ollama
docker compose --profile ollama exec ollama ollama create payee-lora -f /modelfiles/Modelfile
docker compose --profile ollama up --build -d
```

The host Ollama is the default because it can use the host GPU without extra
container configuration; a CPU-only container is several times slower per row.

## Where state lives

| Volume | Holds | Safe to delete? |
|---|---|---|
| `engine-state` | task results (`tasks/`), in-flight uploads, the statement-fingerprint secret, Docling model cache | Deletes all past results; the secret is regenerated |
| `merchant-data` | the **live** merchant dictionary (including auto-promotions), cache, curation ledger | No — it is the curated dictionary |
| `ollama-models` | the containerised payee model | Yes — re-run `ollama create` |

The merchant dictionary in `merchant-data` is seeded from the image the first
time the volume is created. After that it is state: rebuilding the image does
not overwrite it, so auto-promoted merchants survive upgrades. To take a newer
baked dictionary, merge it into the volume's copy and call
`POST /api/v1/merchants/reload`.

## Privacy notes

- Uploaded statements are deleted as soon as they are processed; a statement
  interrupted by a crash is deleted at the next start.
- The engine sends the merchant server only names it could not resolve itself,
  plus an opaque statement fingerprint. Nothing leaves the compose network
  unless `GEMINI_API_KEY` / `LANGSEARCH_API_KEY` are set, and even then only for
  names seen in `MERCHANT_MIN_STATEMENT_COUNT` distinct statements.
- The engine has no authentication. Expose port 8000 only to trusted networks.
