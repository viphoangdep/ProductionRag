"""Thread-safe in-memory task tracker for background corpus ingestion jobs."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Any


class TaskTracker:
    """Quản lý trạng thái và tiến độ của các tác vụ chạy nền."""

    def __init__(self):
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_task(self, total_files: int = 0) -> str:
        """Khởi tạo một task mới ở trạng thái QUEUED."""
        task_id = str(uuid.uuid4())
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "status": "QUEUED",
                "created_at": now,
                "updated_at": now,
                "total_files": total_files,
                "processed_files": 0,
                "current_file": None,
                "result": None,
                "error": None,
            }
        return task_id

    def update_progress(self, task_id: str, processed_files: int, current_file: str | None = None):
        """Cập nhật tiến độ task sang PROCESSING."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task["status"] = "PROCESSING"
                task["processed_files"] = processed_files
                task["current_file"] = current_file
                task["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    def complete_task(self, task_id: str, result: dict[str, Any]):
        """Đánh dấu task đã hoàn thành COMPLETED kèm kết quả."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task["status"] = "COMPLETED"
                task["current_file"] = None
                task["result"] = result
                task["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    def fail_task(self, task_id: str, error: str):
        """Đánh dấu task FAILED kèm thông tin lỗi."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task["status"] = "FAILED"
                task["current_file"] = None
                task["error"] = error
                task["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Lấy thông tin trạng thái của một task."""
        with self._lock:
            return self._tasks.get(task_id)

