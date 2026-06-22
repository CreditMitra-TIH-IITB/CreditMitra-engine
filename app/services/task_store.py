import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


def get_task_file_path(task_id: str) -> Path:
    return Path(settings.DATA_DIR) / "tasks" / f"{task_id}.json"


def save_task(task_id: str, data: dict[str, Any]) -> None:
    """Atomically write task JSON to avoid partial-read race conditions.

    Writes to a temp file in the same directory, then renames it
    over the target. On POSIX this is atomic; on Windows it uses
    os.replace which is as close to atomic as the OS allows.
    """
    path = get_task_file_path(task_id)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=f"{task_id}_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)  # atomic on same filesystem
    except Exception:
        # Clean up the temp file if the rename failed
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def get_task(task_id: str) -> dict[str, Any] | None:
    path = get_task_file_path(task_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            # File exists but is empty (mid-write); treat as not-yet-ready
            logger.debug("Task file %s is empty, returning pending status", task_id)
            return {"task_id": task_id, "status": "processing"}
        result: dict[str, Any] = json.loads(content)
        return result
    except json.JSONDecodeError:
        # File is being written; return a safe fallback
        logger.debug("Task file %s has invalid JSON, returning pending status", task_id)
        return {"task_id": task_id, "status": "processing"}


def update_task_status(task_id: str, status: str, **kwargs: Any) -> None:
    task_data = get_task(task_id)
    if not task_data:
        task_data = {"task_id": task_id}
    task_data["status"] = status
    for k, v in kwargs.items():
        task_data[k] = v
    save_task(task_id, task_data)
