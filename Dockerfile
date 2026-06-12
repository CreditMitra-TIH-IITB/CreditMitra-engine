# ── Build stage ────────────────────────────────────────────
FROM python:3.10-slim-bookworm AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY app/ ./app/

RUN pip install --no-cache-dir --prefix=/install .

# ── Runtime stage ─────────────────────────────────────────
FROM python:3.10-slim-bookworm

# Security: run as non-root
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

# Create runtime data directory with correct ownership
RUN mkdir -p data/tasks data/uploads && chown -R appuser:appuser data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
