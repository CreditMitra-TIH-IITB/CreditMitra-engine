import os

# Must be set before app.core.config is imported: tests should not load the
# payee model or the 2.3GB classifier in the background.
os.environ.setdefault("WARMUP_ON_STARTUP", "false")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    """Provide a FastAPI TestClient for each test."""
    with TestClient(app) as c:
        yield c
