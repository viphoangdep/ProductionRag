"""Middleware to enforce maximum request payload/body size (e.g. 1GB limit)."""
from __future__ import annotations

import logging
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from config.settings import settings

log = logging.getLogger("ai-service")


class PayloadTooLargeException(Exception):
    """Internal exception raised when streaming request body exceeds max size."""


class PayloadSizeLimitMiddleware:
    """Rejects requests exceeding max_size_bytes with HTTP 413 Payload Too Large."""

    def __init__(self, app: ASGIApp, max_size_bytes: int | None = None):
        self.app = app
        self.max_size_bytes = (
            max_size_bytes
            if max_size_bytes is not None
            else getattr(settings, "max_upload_size_bytes", 1024 * 1024 * 1024)
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length_header = headers.get("content-length")

        if content_length_header is not None:
            try:
                content_length = int(content_length_header)
                if content_length > self.max_size_bytes:
                    log.warning(
                        "Payload too large (Content-Length: %d > %d)",
                        content_length,
                        self.max_size_bytes,
                    )
                    response = JSONResponse(
                        {
                            "detail": f"Payload too large. Maximum allowed size is {self.max_size_bytes} bytes."
                        },
                        status_code=413,
                    )
                    await response(scope, receive, send)
                    return
            except (ValueError, TypeError):
                pass

        total_bytes_received = 0

        async def custom_receive():
            nonlocal total_bytes_received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                total_bytes_received += len(body)
                if total_bytes_received > self.max_size_bytes:
                    log.warning(
                        "Streamed payload exceeded limit: %d > %d",
                        total_bytes_received,
                        self.max_size_bytes,
                    )
                    raise PayloadTooLargeException()
            return message

        try:
            await self.app(scope, custom_receive, send)
        except PayloadTooLargeException:
            response = JSONResponse(
                {
                    "detail": f"Payload too large. Maximum allowed size is {self.max_size_bytes} bytes."
                },
                status_code=413,
            )
            await response(scope, receive, send)

