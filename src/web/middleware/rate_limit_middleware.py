"""Lightweight in-memory rate limiter using sliding window algorithm."""
from __future__ import annotations

import time
import logging
from collections import defaultdict
from threading import Lock
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config.settings import settings

log = logging.getLogger("ai-service")


class RateLimiter:
    """Sliding-window rate limiter per client IP."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds

        with self.lock:
            timestamps = self.requests[client_id]
            # Prune timestamps older than cutoff
            valid_timestamps = [t for t in timestamps if t > cutoff]
            self.requests[client_id] = valid_timestamps

            if len(valid_timestamps) >= self.max_requests:
                return False

            self.requests[client_id].append(now)
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int | None = None, window_seconds: int | None = None):
        super().__init__(app)
        max_req = max_requests if max_requests is not None else settings.rate_limit_requests
        win_sec = window_seconds if window_seconds is not None else settings.rate_limit_window_seconds
        self.limiter = RateLimiter(max_requests=max_req, window_seconds=win_sec)

    async def dispatch(self, request: Request, call_next):
        # Exclude health check and doc endpoints from rate limiting
        path = request.url.path
        if path.startswith(("/health", "/docs", "/openapi.json")):
            return await call_next(request)

        # Get client IP or fall back to forwarded headers
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        if not self.limiter.is_allowed(client_ip):
            log.warning("Rate limit exceeded for client_ip=%s path=%s", client_ip, path)
            return JSONResponse(
                {"detail": "Too many requests. Please slow down and try again later."},
                status_code=429,
            )

        return await call_next(request)

