"""Corpus build request models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CorpusBuildRequest(BaseModel):
    """Tham số khi kích hoạt build corpus."""

    recreate_collection: bool = Field(
        default=False,
        description="Xóa collection cũ và tạo mới lại từ đầu trong Qdrant.",
    )
    skip_s3: bool = Field(
        default=False,
        description="Bỏ qua việc tải ảnh và bảng lên AWS S3.",
    )
    only: list[str] = Field(
        default_factory=list,
        description="Chỉ xử lý danh sách các doc_slug được chỉ định.",
    )
    docs_dir: str | None = Field(
        default=None,
        description="Đường dẫn thư mục chứa PDF tuỳ chọn (mặc định lấy theo Settings).",
    )

