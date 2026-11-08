from pathlib import Path
import pytest
from service.corpus_service import (
    CorpusService,
    make_doc_slug,
    make_chunk_id,
    point_id_for,
    heading_info,
)


def test_pure_helpers():
    slug = make_doc_slug(Path("my sample document.pdf"))
    assert slug == "my_sample_document"
    
    cid = make_chunk_id(slug, "v1", 42)
    assert cid == "my_sample_document-v1-0042"
    
    pid = point_id_for(cid)
    assert isinstance(pid, str) and len(pid) == 36

    sp, sid, pid_node, ptitle, anc, depth = heading_info("test.pdf", ["Chap 1", "Sec 1"])
    assert sp == "Chap 1 > Sec 1"
    assert depth == 2
    assert ptitle == "Chap 1"


def test_task_lifecycle_tracking():
    service = CorpusService()
    task_id = service._create_task(total_files=2)
    status = service.get_task_status(task_id)
    assert status["status"] == "QUEUED"
    assert status["total_files"] == 2
    assert status["processed_files"] == 0

    service._update_task_progress(task_id, processed_files=1, current_file="doc1.pdf")
    status = service.get_task_status(task_id)
    assert status["status"] == "PROCESSING"
    assert status["processed_files"] == 1
    assert status["current_file"] == "doc1.pdf"

    service._complete_task(task_id, result={"docs": 2})
    status = service.get_task_status(task_id)
    assert status["status"] == "COMPLETED"
    assert status["result"]["docs"] == 2


def test_task_failure_tracking():
    service = CorpusService()
    task_id = service._create_task(total_files=1)
    service._fail_task(task_id, error="Docling conversion error")
    status = service.get_task_status(task_id)
    assert status["status"] == "FAILED"
    assert "Docling conversion error" in status["error"]

