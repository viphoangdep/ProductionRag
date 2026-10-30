"""Timeout middleware ensuring requests do not hang worker threads indefinitely."""
from __future__ import annotations

import asyncio
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config.settings import settings

log = logging.getLogger("ai-service")


class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout_seconds: int | None = None):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds or settings.request_timeout_seconds

    async def dispatch(self, request: Request, call_next):
        # Exclude documentation or health endpoints if needed
        if request.url.path.startswith(("/health", "/docs", "/openapi.json")):
            return await call_next(request)

        try:
            return await asyncio.wait_for(call_next(request), timeout=float(self.timeout_seconds))
        except asyncio.TimeoutError:
            log.error(
                "Request timeout after %ss method=%s path=%s",
                self.timeout_seconds, request.method, request.url.path
            )
            return JSONResponse(
                {"detail": f"Request processing timed out after {self.timeout_seconds} seconds"},
                status_code=504,
            )

