from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from core.exception.errors import AppError, ErrorCode
from model.req.corpus_req import CorpusBuildRequest
from web.controller.corpus_controller import (
    upload_pdfs,
    trigger_build,
    get_task_status,
    get_manifest,
)


@pytest.fixture
def mock_corpus_service():
    mock = MagicMock()
    mock.start_build_task.return_value = "task-mock-123"
    mock.get_task_status.side_effect = lambda tid: {
        "task_id": tid,
        "status": "PROCESSING",
        "total_files": 1,
        "processed_files": 0,
        "current_file": "test.pdf",
    } if tid == "task-mock-123" else None
    mock.get_manifest.return_value = {
        "total_docs": 1,
        "points_count": 42,
        "manifest": {"test_doc": {"version": "v1"}},
    }
    mock.save_uploaded_files.return_value = [Path("docs/test.pdf")]
    return mock


def test_upload_pdf_triggers_build_task(mock_corpus_service):
    file1 = SimpleNamespace(filename="test.pdf", file=None, content=b"%PDF-1.4 mock")
    res = upload_pdfs(
        files=[file1],
        recreate_collection=False,
        skip_s3=False,
        service=mock_corpus_service,
    )
    assert res["status"] == "queued"
    assert res["task_id"] == "task-mock-123"
    assert res["file_count"] == 1
    assert mock_corpus_service.save_uploaded_files.called
    assert mock_corpus_service.start_build_task.called


def test_upload_non_pdf_rejected(mock_corpus_service):
    file1 = SimpleNamespace(filename="report.docx", file=None, content=b"fake")
    with pytest.raises(AppError) as exc_info:
        upload_pdfs(
            files=[file1],
            recreate_collection=False,
            skip_s3=False,
            service=mock_corpus_service,
        )
    assert exc_info.value.status_code == 400
    assert "PDF" in exc_info.value.detail


def test_trigger_build_endpoint(mock_corpus_service):
    req = CorpusBuildRequest(recreate_collection=True, skip_s3=True, only=["sample_doc"])
    res = trigger_build(body=req, service=mock_corpus_service)
    assert res["status"] == "queued"
    assert res["task_id"] == "task-mock-123"
    mock_corpus_service.start_build_task.assert_called_with(
        recreate_collection=True,
        skip_s3=True,
        only=["sample_doc"],
    )


def test_get_task_status_found(mock_corpus_service):
    res = get_task_status("task-mock-123", service=mock_corpus_service)
    assert res["task_id"] == "task-mock-123"
    assert res["status"] == "PROCESSING"


def test_get_task_status_not_found(mock_corpus_service):
    with pytest.raises(AppError) as exc_info:
        get_task_status("unknown-task-id", service=mock_corpus_service)
    assert exc_info.value.code == ErrorCode.TASK_NOT_FOUND
    assert exc_info.value.status_code == 404


def test_get_manifest_endpoint(mock_corpus_service):
    res = get_manifest(service=mock_corpus_service)
    assert res["total_docs"] == 1
    assert res["points_count"] == 42

