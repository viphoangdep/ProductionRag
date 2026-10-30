import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from core.exception.errors import AppError

log = logging.getLogger("ai-service")


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Lỗi nghiệp vụ -> status_code của nó (404, 500...)."""
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


async def app_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled %s %s", request.method, request.url.path)
    return JSONResponse({"detail": "internal error"}, status_code=500)
