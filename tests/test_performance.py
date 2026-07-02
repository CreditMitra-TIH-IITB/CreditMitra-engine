import pytest
import time
import os
from fastapi.testclient import TestClient

def test_upload_performance_large_file(client):
    """Test performance of the file upload endpoint."""
    # Create a large dummy PDF file (10MB)
    size = 10 * 1024 * 1024
    content = b"0" * size

    start_time = time.time()

    # Upload the file
    response = client.post(
        "/api/v1/statements/process",
        files={"pdf": ("report.pdf", content, "application/pdf")},
    )

    end_time = time.time()

    assert response.status_code == 200

    print(f"\\nUpload time for 10MB: {end_time - start_time:.4f} seconds")
