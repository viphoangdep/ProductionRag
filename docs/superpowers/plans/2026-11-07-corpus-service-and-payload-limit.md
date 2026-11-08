# Corpus Service, Full API & 1GB Payload Limit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `scripts/build_corpus.py` into a robust `CorpusService` in `src/service/`, expose a complete REST API with background task execution, protect the application with a 1GB payload limit middleware, and preserve CLI compatibility.

**Architecture:** 
- A custom ASGI `PayloadSizeLimitMiddleware` intercepts requests and enforces the 1GB payload ceiling (both for `Content-Length` headers and streaming bodies).
- A domain `CorpusService` encapsulates Docling conversion, hierarchical chunking, S3 upload, Qdrant streaming upsert, manifest merging, and in-memory background task status tracking.
- A `CorpusController` exposes endpoints (`/upload`, `/build`, `/tasks/{task_id}`, `/manifest`) wired via `Container` and `deps.py`.
- `scripts/build_corpus.py` becomes a thin CLI wrapper delegating to `CorpusService.build_corpus_sync()`.

**Tech Stack:** Python 3.12, FastAPI, Starlette ASGI, Pydantic v2, Dependency-Injector, Qdrant Client, Docling, LangChain, Pytest.

**Spec:** [Tài liệu Thiết Kế: Corpus Management Service, API & 1GB Payload Limit Middleware](docs/superpowers/plans/2026-11-07-corpus-service-and-payload-limit.md#design-spec)

---

## Global Constraints

- Payload limit: 1GB (`1024 * 1024 * 1024` bytes) configurable in `Settings.max_upload_size_bytes`.
- Error handling: Domain errors must raise `AppError(ErrorCode.XYZ)` mapped to proper HTTP status codes.
- Timeout safety: Heavy document processing must run asynchronously in background tasks to avoid violating `TimeoutMiddleware` (60s).
- Compatibility: `python scripts/build_corpus.py` CLI arguments and behavior must remain functional.
- Test execution: All tests run via `python -m pytest`.

---

## Review Focus

1. **Streamed payload without `Content-Length` exceeds 1GB**: Middleware must count bytes during chunk consumption and abort with HTTP 413 without reading the entire payload into RAM.
2. **Non-PDF file upload**: Upload endpoint must validate file extension (`.pdf`), rejecting invalid files with HTTP 400.
3. **Task failure resilience**: If Docling conversion or Qdrant upsert fails during a background task, the task state must transition to `FAILED` with error details rather than crashing the process.
4. **Missing AWS S3 credentials**: Pipeline must gracefully skip S3 upload when credentials are absent or `skip_s3=True` without throwing unhandled exceptions.
5. **Polled task not found**: Querying `/tasks/{unknown_id}` must return HTTP 404 (`ErrorCode.TASK_NOT_FOUND`).

---

### Task 1: Payload Size Limit Middleware & Domain Error Code

**Files:**
- Modify: `src/core/exception/errors.py`
- Create: `src/web/middleware/payload_size_middleware.py`
- Test: `tests/test_payload_size_middleware.py`

**Interfaces:**
- Consumes: `AppError`, `ErrorCode`
- Produces: `PayloadSizeLimitMiddleware(app, max_size_bytes: int = 1024 * 1024 * 1024)`

- [x] **Step 1: Write failing test for `PayloadSizeLimitMiddleware`**

```python
# tests/test_payload_size_middleware.py
import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from web.middleware.payload_size_middleware import PayloadSizeLimitMiddleware


async def ok_endpoint(request):
    data = await request.body()
    return JSONResponse({"received_bytes": len(data)})


@pytest.fixture
def app():
    routes = [Route("/upload", ok_endpoint, methods=["POST"])]
    # Use 100 bytes limit for fast testing
    middleware = [Middleware(PayloadSizeLimitMiddleware, max_size_bytes=100)]
    return Starlette(routes=routes, middleware=middleware)


def test_request_under_limit_succeeds(app):
    client = TestClient(app)
    response = client.post("/upload", content=b"a" * 50)
    assert response.status_code == 200
    assert response.json()["received_bytes"] == 50


def test_request_with_large_content_length_header_rejected(app):
    client = TestClient(app)
    headers = {"Content-Length": "1000"}
    response = client.post("/upload", content=b"a" * 50, headers=headers)
    assert response.status_code == 413
    assert "Payload too large" in response.json()["detail"]


def test_streamed_request_exceeding_limit_rejected(app):
    client = TestClient(app)

    def stream_large_data():
        for _ in range(3):
            yield b"a" * 50  # Total 150 bytes > 100 bytes limit

    response = client.post("/upload", content=stream_large_data())
    assert response.status_code == 413
    assert "Payload too large" in response.json()["detail"]
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_payload_size_middleware.py`
Expected: FAIL (cannot import `PayloadSizeLimitMiddleware`)

- [x] **Step 3: Update `src/core/exception/errors.py` and implement `PayloadSizeLimitMiddleware`**

Add `ErrorCode.PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"` mapped to `(413, "Payload too large. Maximum allowed size is 1GB.")` in `src/core/exception/errors.py`.
Add `ErrorCode.TASK_NOT_FOUND = "TASK_NOT_FOUND"` mapped to `(404, "Corpus task not found")`.
Implement `PayloadSizeLimitMiddleware` in `src/web/middleware/payload_size_middleware.py` checking `Content-Length` header and wrapping `receive` to track streaming bytes, returning a JSON 413 response if exceeded.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_payload_size_middleware.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/core/exception/errors.py src/web/middleware/payload_size_middleware.py tests/test_payload_size_middleware.py
git commit -m "feat(middleware): add PayloadSizeLimitMiddleware with 1GB default limit"
```

---

### Task 2: Corpus Settings & Request/Response Models

**Files:**
- Modify: `src/config/settings.py`
- Create: `src/model/req/corpus_req.py`
- Create: `src/model/res/corpus_res.py`
- Test: `tests/test_corpus_models.py`

**Interfaces:**
- Produces: `CorpusBuildRequest`, `CorpusTaskResponse`, `CorpusManifestResponse`, updated `Settings`

- [x] **Step 1: Write failing test for corpus configuration and request/response models**

```python
# tests/test_corpus_models.py
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus_models.py`
Expected: FAIL (cannot import `CorpusBuildRequest` / `CorpusTaskResponse`)

- [x] **Step 3: Update `Settings` and create models**

- In `src/config/settings.py`: Add `max_upload_size_bytes: int = 1024 * 1024 * 1024`, `docs_dir: str = "docs"`, `embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"`, `chunk_max_tokens: int = 512`, `text_batch: int = 4`, `skip_s3: bool = False`, S3 env settings, and `@property def docs_path(self) -> Path`.
- In `src/model/req/corpus_req.py`: Implement `CorpusBuildRequest`.
- In `src/model/res/corpus_res.py`: Implement `CorpusTaskResponse` and `CorpusManifestResponse`.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_corpus_models.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/config/settings.py src/model/req/corpus_req.py src/model/res/corpus_res.py tests/test_corpus_models.py
git commit -m "feat(corpus): add corpus settings and request/response models"
```

---

### Task 3: CorpusService Implementation & Pure Helpers

**Files:**
- Create: `src/service/corpus_service.py`
- Test: `tests/test_corpus_service.py`

**Interfaces:**
- Consumes: `Settings`, `CorpusBuildRequest`, `CorpusTaskResponse`
- Produces: `CorpusService` with methods:
  - `save_uploaded_files(files) -> list[Path]`
  - `start_build_task(pdf_paths=None, recreate_collection=False, skip_s3=False, only=None) -> str`
  - `get_task_status(task_id: str) -> dict`
  - `build_corpus_sync(docs_dir=None, output_dir=None, recreate_collection=False, skip_s3=False, only=None) -> dict`
  - `get_manifest() -> dict`

- [x] **Step 1: Write unit tests for pure helpers and `CorpusService` task tracking**

```python
# tests/test_corpus_service.py
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
    # Dummy task
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus_service.py`
Expected: FAIL (cannot import `CorpusService`)

- [x] **Step 3: Implement `src/service/corpus_service.py`**

Port pure helpers from `scripts/build_corpus.py` (`make_doc_slug`, `make_chunk_id`, `point_id_for`, `make_section_path`, `make_node_id`, `normalize_headings`, `heading_info`, `get_chunk_pages`, `get_chunk_page_boxes`, `vertical_distance`, `chunk_body`, `embed_with_retry`).
Implement `CorpusService`:
- `__init__(self, settings=settings)`
- Task tracking dictionary `self._tasks: dict[str, dict]` protected with threading lock.
- Helper methods `_create_task`, `_update_task_progress`, `_complete_task`, `_fail_task`.
- Client builders: `build_converter`, `build_chunker`, `build_s3`, `ensure_collection`, `get_existing_ids`.
- Core pipeline methods: `process_one_pdf`, `merge_outputs`, `build_corpus_sync`.
- Async background wrapper: `start_build_task` launching a background worker thread.
- `save_uploaded_files(self, files: list) -> list[Path]`.
- `get_manifest(self) -> dict`.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_corpus_service.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/service/corpus_service.py tests/test_corpus_service.py
git commit -m "feat(corpus): implement CorpusService with pipeline and task tracking"
```

---

### Task 4: Web Controller, Container Wiring & API Integration

**Files:**
- Create: `src/web/controller/corpus_controller.py`
- Modify: `src/core/container.py`
- Modify: `src/web/deps.py`
- Modify: `src/main.py`
- Test: `tests/test_corpus_api.py`

**Interfaces:**
- Produces: API routes under `/api/v1/corpus`:
  - `POST /upload`
  - `POST /build`
  - `GET /tasks/{task_id}`
  - `GET /manifest`
- Container: `container.corpus_service`
- Deps: `deps.get_corpus_service`

- [x] **Step 1: Write integration tests for Corpus API endpoints**

```python
# tests/test_corpus_api.py
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from core.database.assets import AssetService
from main import create_app
from web import deps


@asynccontextmanager
async def _no_lifespan(app):
    yield


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
    mock.get_manifest.return_value = {"docs": {"test_doc": {"version": "v1"}}, "points_count": 42}
    mock.save_uploaded_files.return_value = [Path("docs/test.pdf")]
    return mock


@pytest.fixture
def client(mock_corpus_service):
    app = create_app()
    app.router.lifespan_context = _no_lifespan
    app.dependency_overrides[deps.get_asset_store] = lambda: AssetService(Path("."))
    app.dependency_overrides[deps.get_corpus_service] = lambda: mock_corpus_service
    return TestClient(app, raise_server_exceptions=False)


def test_upload_pdf_triggers_build_task(client, mock_corpus_service):
    files = [("files", ("test.pdf", b"%PDF-1.4 mock content", "application/pdf"))]
    r = client.post("/api/v1/corpus/upload", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["task_id"] == "task-mock-123"
    assert data["status"] == "queued"
    assert mock_corpus_service.save_uploaded_files.called


def test_upload_invalid_file_rejected(client):
    files = [("files", ("test.txt", b"not a pdf", "text/plain"))]
    r = client.post("/api/v1/corpus/upload", files=files)
    assert r.status_code == 400
    assert "PDF" in r.json()["detail"]


def test_build_endpoint(client, mock_corpus_service):
    r = client.post("/api/v1/corpus/build", json={"recreate_collection": False})
    assert r.status_code == 200
    assert r.json()["task_id"] == "task-mock-123"


def test_get_task_status(client):
    r = client.get("/api/v1/corpus/tasks/task-mock-123")
    assert r.status_code == 200
    assert r.json()["status"] == "PROCESSING"


def test_get_unknown_task_returns_404(client):
    r = client.get("/api/v1/corpus/tasks/unknown-task")
    assert r.status_code == 404
    assert r.json()["detail"] == "Corpus task not found"


def test_get_manifest(client):
    r = client.get("/api/v1/corpus/manifest")
    assert r.status_code == 200
    assert r.json()["points_count"] == 42
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus_api.py`
Expected: FAIL (router not registered or endpoints missing)

- [x] **Step 3: Implement controller, container provider, deps and register middleware in `main.py`**

- In `src/web/controller/corpus_controller.py`: Implement `router = APIRouter(prefix="/api/v1/corpus", tags=["corpus"])` with `/upload`, `/build`, `/tasks/{task_id}`, `/manifest`.
- In `src/core/container.py`: Register `corpus_service = providers.Singleton(CorpusService)`.
- In `src/web/deps.py`: Add `get_corpus_service() -> CorpusService`.
- In `src/main.py`: Add `PayloadSizeLimitMiddleware(max_size_bytes=settings.max_upload_size_bytes)` and include `corpus_router`.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_corpus_api.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/web/controller/corpus_controller.py src/core/container.py src/web/deps.py src/main.py tests/test_corpus_api.py
git commit -m "feat(api): add CorpusController and wire into FastAPI application"
```

---

### Task 5: Refactor `scripts/build_corpus.py` CLI Wrapper

**Files:**
- Modify: `scripts/build_corpus.py`
- Test: `tests/test_build_corpus_cli.py`

**Interfaces:**
- Consumes: `CorpusService`
- Produces: CLI interface `python scripts/build_corpus.py [--docs ...] [--output ...] [--version ...] [--recreate-collection] [--skip-s3] [--only ...]`

- [x] **Step 1: Write test for `scripts/build_corpus.py` CLI invocation**

```python
# tests/test_build_corpus_cli.py
from unittest.mock import patch, MagicMock
from scripts.build_corpus import main


@patch("scripts.build_corpus.CorpusService")
def test_cli_invokes_service_build_sync(mock_service_cls):
    mock_inst = MagicMock()
    mock_inst.build_corpus_sync.return_value = {"doc1": {}}
    mock_service_cls.return_value = mock_inst

    code = main(["--docs", "docs", "--version", "v2", "--skip-s3"])
    assert code == 0
    assert mock_inst.build_corpus_sync.called
    kwargs = mock_inst.build_corpus_sync.call_args.kwargs
    assert kwargs["version"] == "v2"
    assert kwargs["skip_s3"] is True
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_corpus_cli.py`
Expected: FAIL

- [x] **Step 3: Refactor `scripts/build_corpus.py`**

Refactor `scripts/build_corpus.py` to keep argument parsing and delegate the heavy execution to `CorpusService().build_corpus_sync(...)`. Keep helpers importable or re-exported for backwards compatibility.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_build_corpus_cli.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add scripts/build_corpus.py tests/test_build_corpus_cli.py
git commit -m "refactor(scripts): turn build_corpus.py into thin CLI calling CorpusService"
```

---

### Task 6: Full Verification & Integration Check

**Files:**
- Test: Run full test suite covering all existing and new tests.

- [x] **Step 1: Run all unit and integration tests**

Run: `python -m pytest tests/test_payload_size_middleware.py tests/test_corpus_models.py tests/test_corpus_service.py tests/test_corpus_api.py tests/test_build_corpus_cli.py tests/test_chat_service.py -v`
Expected: All tests PASS.

- [x] **Step 2: Verify git status and clean working tree**

Run: `git status`
Expected: Working tree clean.

