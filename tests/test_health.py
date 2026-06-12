"""Smoke tests for core API endpoints."""


def test_health_returns_ok(client):
    """GET /health should return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_upload_rejects_non_pdf(client):
    """POST /api/v1/statements/process should reject non-PDF files."""
    response = client.post(
        "/api/v1/statements/process",
        files={"pdf": ("report.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_status_returns_404_for_unknown_task(client):
    """GET /api/v1/statements/status/{id} should 404 for non-existent task."""
    response = client.get("/api/v1/statements/status/non-existent-id")
    assert response.status_code == 404
