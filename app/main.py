import faulthandler
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import statements, system
from app.core.config import settings
from app.services.task_store import (
    check_encryption_key,
    fail_interrupted_tasks,
    purge_expired_results,
)
from app.services.warmup import start_warmup

# Uvicorn configures only its own loggers, so the engine's INFO messages (task
# progress, warm-up) were dropped. Attach a handler to the `app` namespace only:
# third-party INFO (httpx logs every Ollama call, Docling every page) stays off.
_app_logger = logging.getLogger("app")
if not _app_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _app_logger.addHandler(_handler)
_app_logger.setLevel(settings.LOG_LEVEL)

# Docling (torch) and the classifier (onnxruntime) are native code; a crash in
# either kills the process with no Python traceback. Dump one to stderr.
faulthandler.enable()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    check_encryption_key()
    purged = purge_expired_results()
    if purged:
        _app_logger.info(f"Deleted {purged} task result(s) past the retention period")
    recovered = fail_interrupted_tasks()
    if recovered:
        _app_logger.warning(f"Marked {recovered} interrupted task(s) as failed")
    start_warmup()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Backend API for Credit Mitra Statement Processing",
    version="1.0.0",
    lifespan=lifespan,
)

# Set up CORS for the React frontend. Origins come from CORS_ORIGINS so that a
# LAN or Tauri host can be added without widening this to a wildcard: Starlette
# echoes the caller's origin back, so a permissive regex plus credentials would
# let any site make credentialed calls to a user's local engine.
_cors_kwargs: dict[str, Any] = {
    "allow_origins": settings.cors_origin_list,
    "allow_credentials": False,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if settings.CORS_ORIGIN_REGEX:
    _cors_kwargs["allow_origin_regex"] = settings.CORS_ORIGIN_REGEX

app.add_middleware(CORSMiddleware, **_cors_kwargs)

app.include_router(
    statements.router, prefix=f"{settings.API_V1_STR}/statements", tags=["statements"]
)
app.include_router(
    system.router, prefix=f"{settings.API_V1_STR}/system", tags=["system"]
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
