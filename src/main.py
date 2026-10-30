"""FastAPI entrypoint. Mọi wiring nằm ở web/deps.py — main chỉ lắp router + middleware."""
import os

# Suppress trước mọi import ML: TF/oneDNN/absl/tokenizers ồn lúc import CrossEncoder.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*")

import logging
# uvicorn chỉ config logger của nó; không có dòng này thì log INFO của
# "ai-service" lan lên root (không handler) rồi rơi mất -> không thấy gì.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.exception.errors import AppError
from web.controller.chat_controller import router as chat_router
from web.controller.health_controller import router as health_router
from web.middleware.exception_handler import app_error_handler, app_exception_handler
from web.middleware.logging_middleware import LoggingMiddleware
from web.middleware.rate_limit_middleware import RateLimitMiddleware
from web.middleware.timeout_middleware import TimeoutMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    from web import deps  # lazy: sau khi env suppress đã set

    deps.warmup()  # fail-fast: lỗi là crash để biết đường fix
    try:
        yield
    finally:
        deps.close()  # đóng client vector store


def create_app() -> FastAPI:
    app = FastAPI(title="ai-service", lifespan=lifespan)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TimeoutMiddleware)
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, app_exception_handler)
    app.include_router(chat_router)
    app.include_router(health_router)
    return app


app = create_app()
