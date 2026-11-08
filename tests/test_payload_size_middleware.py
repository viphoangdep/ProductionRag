import json
import pytest
from web.middleware.payload_size_middleware import PayloadSizeLimitMiddleware


def run_coro(coro):
    """Drive ASGI coroutines deterministically without socket/loop dependencies."""
    try:
        while True:
            coro.send(None)
    except StopIteration:
        pass


def test_request_under_limit_succeeds():
    sent = []

    async def dummy_app(scope, receive, send):
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"received": ' + str(len(body)).encode() + b"}"})

    mw = PayloadSizeLimitMiddleware(dummy_app, max_size_bytes=100)
    scope = {"type": "http", "method": "POST", "path": "/upload", "headers": [(b"content-length", b"50")]}

    async def mock_recv():
        return {"type": "http.request", "body": b"a" * 50, "more_body": False}

    async def mock_send(m):
        sent.append(m)

    run_coro(mw(scope, mock_recv, mock_send))
    assert len(sent) >= 2
    assert sent[0]["status"] == 200


def test_request_with_large_content_length_header_rejected():
    sent = []

    async def dummy_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = PayloadSizeLimitMiddleware(dummy_app, max_size_bytes=100)
    scope = {"type": "http", "method": "POST", "path": "/upload", "headers": [(b"content-length", b"150")]}

    async def mock_recv():
        return {"type": "http.request", "body": b"a" * 50, "more_body": False}

    async def mock_send(m):
        sent.append(m)

    run_coro(mw(scope, mock_recv, mock_send))
    assert len(sent) >= 2
    assert sent[0]["status"] == 413
    body = json.loads(sent[1]["body"].decode())
    assert "Payload too large" in body["detail"]


def test_streamed_request_exceeding_limit_rejected():
    sent = []
    chunks = [
        {"type": "http.request", "body": b"a" * 60, "more_body": True},
        {"type": "http.request", "body": b"a" * 60, "more_body": False},
    ]

    async def mock_recv():
        return chunks.pop(0)

    async def mock_send(m):
        sent.append(m)

    async def dummy_app_read(scope, receive, send):
        await receive()
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = PayloadSizeLimitMiddleware(dummy_app_read, max_size_bytes=100)
    scope = {"type": "http", "method": "POST", "path": "/upload", "headers": []}

    run_coro(mw(scope, mock_recv, mock_send))
    assert len(sent) >= 2
    assert sent[0]["status"] == 413
    body = json.loads(sent[1]["body"].decode())
    assert "Payload too large" in body["detail"]
