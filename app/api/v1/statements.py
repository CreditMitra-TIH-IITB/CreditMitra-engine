import asyncio
import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.core.config import settings
from app.schemas.statements import TaskResponse, TaskStatusResponse
from app.services.extraction import process_pdf_task
from app.services.task_store import get_task, update_task_status

router = APIRouter()


@router.post("/process", response_model=TaskResponse)
async def upload_statement(
    background_tasks: BackgroundTasks,
    pdf: Annotated[UploadFile, File(...)],
) -> TaskResponse:
    """Upload a PDF statement for background processing."""
    filename = pdf.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    task_id = str(uuid.uuid4())

    # Save file to disk temporarily
    suffix = Path(filename).suffix
    file_path = os.path.join(settings.DATA_DIR, "uploads", f"{task_id}{suffix}")

    content = await pdf.read()

    def write_file(path: str, data: bytes) -> None:
        with open(path, "wb") as f:
            f.write(data)

    await asyncio.to_thread(write_file, file_path, content)

    # Initialize task state
    await asyncio.to_thread(update_task_status, task_id, "pending")

    # Dispatch background job
    background_tasks.add_task(process_pdf_task, task_id, file_path)

    return TaskResponse(
        task_id=task_id,
        status="pending",
        message="Statement uploaded successfully. Processing started in background.",
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_processing_status(task_id: str) -> TaskStatusResponse:
    """Check the status of a background processing task."""
    task_data = await asyncio.to_thread(get_task, task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(**task_data)
