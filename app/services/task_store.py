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
