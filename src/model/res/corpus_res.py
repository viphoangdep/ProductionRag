"""Corpus build and task response models."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class CorpusTaskResponse(BaseModel):
    """Thông tin trạng thái task build corpus."""

    task_id: str = Field(description="Mã định danh duy nhất của task build.")
    status: Literal["QUEUED", "PROCESSING", "COMPLETED", "FAILED"] = Field(
        description="Trạng thái hiện tại của task.",
    )
    created_at: str = Field(description="Thời điểm khởi tạo task (ISO format).")
    updated_at: str | None = Field(default=None, description="Thời điểm cập nhật gần nhất.")
    total_files: int = Field(default=0, description="Tổng số file PDF cần xử lý.")
    processed_files: int = Field(default=0, description="Số file PDF đã xử lý xong.")
    current_file: str | None = Field(default=None, description="File PDF đang được xử lý.")
    result: dict[str, Any] | None = Field(default=None, description="Kết quả chi tiết khi task hoàn thành.")
    error: str | None = Field(default=None, description="Thông báo lỗi nếu task thất bại.")


class CorpusManifestResponse(BaseModel):
    """Thông tin manifest hiện có của corpus."""

    total_docs: int = Field(description="Số tài liệu đã được index.")
    points_count: int | None = Field(default=None, description="Số điểm vector trong Qdrant.")
    manifest: dict[str, Any] = Field(default_factory=dict, description="Chi tiết manifest tài liệu.")

