import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("ai-service")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        t = time.time()
        resp = await call_next(request)
        log.info("%s %s -> %s %.0fms", request.method, request.url.path,
                 resp.status_code, (time.time() - t) * 1000)
        return resp
