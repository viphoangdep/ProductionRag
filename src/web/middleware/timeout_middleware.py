"""Timeout middleware ensuring requests do not hang worker threads indefinitely."""
from __future__ import annotations

import asyncio
import logging
from starlette.responses import JSONResponse

from config.settings import settings

log = logging.getLogger("ai-service")


class TimeoutMiddleware:
    """Pure ASGI timeout middleware to prevent worker thread hang without BaseHTTPMiddleware deadlocks."""

    def __init__(self, app, timeout_seconds: int | None = None):
        self.app = app
        self.timeout_seconds = timeout_seconds or settings.request_timeout_seconds

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path.startswith(("/health", "/docs", "/openapi.json")):
            await self.app(scope, receive, send)
            return

        timeout = float(self.timeout_seconds)
        response_started = False

        async def send_wrapper(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            async with asyncio.timeout(timeout):
                await self.app(scope, receive, send_wrapper)
        except TimeoutError:
            if not response_started:
                method = scope.get("method", "")
                log.error("Request timeout after %ss method=%s path=%s", self.timeout_seconds, method, path)
                response = JSONResponse(
                    {"detail": f"Request processing timed out after {self.timeout_seconds} seconds"},
                    status_code=504,
                )
                await response(scope, receive, send)
            else:
                log.error("Request timeout after response started: %s", path)


