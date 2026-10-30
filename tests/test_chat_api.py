"""Endpoint: AppError -> status_code đúng (404/500), không phải 500 chung."""
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.database.assets import AssetService
from core.exception.errors import AppError, ErrorCode
from model.req.chat_req import ChatQueryRequest
from web import deps


@asynccontextmanager
async def _no_lifespan(app):
    yield


@pytest.fixture
def client():
    from main import create_app  # import trong fixture: main set env trước khi import ML

    app = create_app()
    app.router.lifespan_context = _no_lifespan
    app.dependency_overrides[deps.get_asset_store] = lambda: AssetService(Path("."))
    return TestClient(app, raise_server_exceptions=False)


def test_unknown_image_returns_404(client):
    r = client.get("/api/v1/chat/assets/images/does-not-exist")
    assert r.status_code == 404
    assert r.json()["detail"] == "image not found"


def test_business_error_keeps_its_status(client):
    class Boom:
        def query(self, body):
            raise AppError(ErrorCode.AGENT_TIMEOUT, "too many rounds")

    app_client = client
    app_client.app.dependency_overrides[deps.get_chat_service] = lambda: Boom()
    r = app_client.post("/api/v1/chat/query", json=ChatQueryRequest(query="q").model_dump())
    assert r.status_code == 504
    assert r.json()["detail"] == "too many rounds"


def test_delete_session_endpoint(client):
    class DummyService:
        def reset_session(self, sid):
            return True

    app_client = client
    app_client.app.dependency_overrides[deps.get_chat_service] = lambda: DummyService()
    r = app_client.delete("/api/v1/chat/sessions/sess-123")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "session_id": "sess-123", "existed": True}
