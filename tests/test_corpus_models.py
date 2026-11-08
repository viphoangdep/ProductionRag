from config.settings import settings
from model.req.corpus_req import CorpusBuildRequest
from model.res.corpus_res import CorpusTaskResponse, CorpusManifestResponse


def test_corpus_settings_defaults():
    assert settings.max_upload_size_bytes == 1024 * 1024 * 1024
    assert settings.docs_dir == "docs"
    assert settings.docs_path.exists() or settings.docs_path.is_absolute()
    assert settings.chunk_max_tokens == 512
    assert settings.text_batch == 4


def test_corpus_build_request_defaults():
    req = CorpusBuildRequest()
    assert req.recreate_collection is False
    assert req.skip_s3 is False
    assert req.only == []


def test_corpus_task_response_structure():
    res = CorpusTaskResponse(
        task_id="test-123",
        status="QUEUED",
        created_at="2026-11-07T00:00:00Z",
        total_files=2,
        processed_files=0,
    )
    assert res.task_id == "test-123"
    assert res.status == "QUEUED"

