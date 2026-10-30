"""Getters cho FastAPI Depends. Bean build ở core/container.py.

Endpoint chỉ Depends các hàm này — không tự build bean nào.
Test override qua app.dependency_overrides.
"""
from core.container import container
from core.container import close as close_container
from core.container import warmup as warmup_container

_WARMUP = {"state": "cold", "detail": ""}


def get_chat_service():
    return container.chat_service()


def get_asset_store():
    return container.asset_store()


def warmup() -> dict:
    """Build toàn bộ bean lúc startup. Fail-fast: lỗi là raise để crash."""
    global _WARMUP
    _WARMUP = {"state": "warming", "detail": ""}
    _WARMUP = warmup_container()
    return _WARMUP


def close() -> None:
    """Đóng resource (client vector store) khi app tắt."""
    close_container()
