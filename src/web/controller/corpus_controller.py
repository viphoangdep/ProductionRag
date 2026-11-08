"""Corpus API controller: upload tài liệu, kích hoạt build, theo dõi task và xem manifest."""
from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, Depends, File, Form, UploadFile

from core.exception.errors import AppError, ErrorCode
from model.req.corpus_req import CorpusBuildRequest
from model.res.corpus_res import CorpusManifestResponse, CorpusTaskResponse
from web import deps

log = logging.getLogger("ai-service")

router = APIRouter(prefix="/api/v1/corpus", tags=["corpus"])


@router.post("/upload")
def upload_pdfs(
    files: list[UploadFile] = File(..., description="Danh sách file PDF cần index"),
    recreate_collection: bool = Form(default=False, description="Xóa và tạo lại Qdrant collection"),
    skip_s3: bool = Form(default=False, description="Bỏ qua upload ảnh/bảng lên S3"),
    service=Depends(deps.get_corpus_service),
) -> dict[str, Any]:
    """Upload các file PDF và kích hoạt background task build corpus."""
    if not files:
        raise AppError(ErrorCode.NOT_FOUND, detail="No files uploaded.")

    # Validate PDF format
    for f in files:
        name = getattr(f, "filename", "") or ""
        if not name.lower().endswith(".pdf"):
            raise AppError(
                ErrorCode.PROMPT_INJECTION_DETECTED,  # 400 Bad Request
                detail=f"File '{name}' không đúng định dạng PDF. Chỉ chấp nhận các file có đuôi .pdf.",
            )

    saved_paths = service.save_uploaded_files(files)
    task_id = service.start_build_task(
        pdf_paths=saved_paths,
        recreate_collection=recreate_collection,
        skip_s3=skip_s3,
    )
    return {
        "status": "queued",
        "task_id": task_id,
        "file_count": len(saved_paths),
        "detail": "File upload thành công, tiến trình build corpus đang chạy ngầm.",
    }


@router.post("/build")
def trigger_build(
    body: CorpusBuildRequest,
    service=Depends(deps.get_corpus_service),
) -> dict[str, Any]:
    """Kích hoạt build corpus từ các file PDF sẵn có trong thư mục docs."""
    task_id = service.start_build_task(
        recreate_collection=body.recreate_collection,
        skip_s3=body.skip_s3,
        only=body.only or None,
    )
    return {
        "status": "queued",
        "task_id": task_id,
        "detail": "Tiến trình build corpus đã được xếp hàng xử lý.",
    }


@router.get("/tasks/{task_id}")
def get_task_status(
    task_id: str,
    service=Depends(deps.get_corpus_service),
) -> dict[str, Any]:
    """Xem tiến độ và trạng thái chi tiết của một build task."""
    status = service.get_task_status(task_id)
    if status is None:
        raise AppError(ErrorCode.TASK_NOT_FOUND, detail=f"Corpus task '{task_id}' không tồn tại.")
    return status


@router.get("/manifest")
def get_manifest(
    service=Depends(deps.get_corpus_service),
) -> dict[str, Any]:
    """Xem manifest tài liệu đã index và số lượng vector hiện có trong Qdrant."""
    return service.get_manifest()

