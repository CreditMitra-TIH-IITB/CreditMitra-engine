import json
from pathlib import Path
from typing import Any

from app.core.config import settings


def get_task_file_path(task_id: str) -> Path:
    return Path(settings.DATA_DIR) / "tasks" / f"{task_id}.json"


def save_task(task_id: str, data: dict[str, Any]) -> None:
    path = get_task_file_path(task_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_task(task_id: str) -> dict[str, Any] | None:
    path = get_task_file_path(task_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
        return result


def update_task_status(task_id: str, status: str, **kwargs: Any) -> None:
    task_data = get_task(task_id)
    if not task_data:
        task_data = {"task_id": task_id}
    task_data["status"] = status
    for k, v in kwargs.items():
        task_data[k] = v
    save_task(task_id, task_data)


INTERRUPTED_ERROR = (
    "The engine stopped before this statement finished processing. Please upload it again."
)


def fail_interrupted_tasks() -> int:
    """Mark tasks left pending/processing by a previous process as failed.

    Tasks run inside the engine process, so at startup any task still in
    flight died with the last one (a crash, a kill, a restart). Left alone it
    stays "processing" forever and the dashboard polls it indefinitely. Its
    upload — a bank statement the crash kept from being cleaned up — is deleted
    too. Uploads belonging to finished tasks are not touched.
    """
    tasks_dir = Path(settings.DATA_DIR) / "tasks"
    uploads_dir = Path(settings.DATA_DIR) / "uploads"
    recovered = 0
    for path in tasks_dir.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if data.get("status") not in ("pending", "processing"):
            continue
        task_id = path.stem
        data["status"] = "failed"
        data["error"] = INTERRUPTED_ERROR
        save_task(task_id, data)
        for upload in uploads_dir.glob(f"{task_id}.*"):
            upload.unlink(missing_ok=True)
        recovered += 1
    return recovered
