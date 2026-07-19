import asyncio
import os
import uuid
from pathlib import Path
from typing import Annotated

import anyio
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.core.config import settings
from app.schemas.statements import ReportResponse, TaskResponse, TaskStatusResponse
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
    async with await anyio.open_file(file_path, "wb") as f:
        await f.write(content)

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
    """Extraction page: task status + transactions with payee/merchant
    classification. Never carries the lifestyle report — see /report."""
    task_data = await asyncio.to_thread(get_task, task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(**task_data)


@router.get("/report/{task_id}", response_model=ReportResponse)
async def get_report(task_id: str) -> ReportResponse:
    """Report page: just the lifestyle/credit-risk report for a task.

    404 only when the task itself doesn't exist. A task that exists but
    hasn't finished (or finished with scoring silently failed — Issue #14)
    returns 200 with `report: null`; check `status` to tell those apart.
    """
    task_data = await asyncio.to_thread(get_task, task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    return ReportResponse(**task_data)
