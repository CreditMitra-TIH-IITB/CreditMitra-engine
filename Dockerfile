# ── Build stage ────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# CPU-only torch first. Docling depends on torch, and the default wheel pulls
# ~2.5 GB of CUDA libraries a CPU container never uses.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app/ ./app/
RUN pip install --no-cache-dir ".[llm]"

# ── Runtime stage ─────────────────────────────────────────
FROM python:3.11-slim-bookworm

# OpenMP runtime for onnxruntime and torch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Security: run as non-root
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

COPY --from=builder /opt/venv /opt/venv

# /app/data   — read-only reference data baked into the image (merchant
#               dictionary, person-name list) plus the classifier models,
#               which are too large for git and are mounted at runtime.
# /app/state  — everything the engine writes: task results, uploads (deleted
#               after processing), the statement-fingerprint secret, and the
#               Docling/Hugging Face model cache. Mount a volume here.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/state \
    HF_HOME=/app/state/hf-cache

WORKDIR /app
COPY app/ ./app/
COPY data/india_merchants.json data/person_name_tokens.txt ./data/

RUN mkdir -p data/models state \
    && chown -R appuser:appuser data state

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
