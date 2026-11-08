"""Lỗi nghiệp vụ: domain error codes và AppError base class."""
from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Mã lỗi miền nghiệp vụ (domain error codes)."""

    NOT_FOUND = "NOT_FOUND"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    AGENT_FAILED = "AGENT_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
    TOOL_BUDGET_EXCEEDED = "TOOL_BUDGET_EXCEEDED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    CORPUS_BUILD_FAILED = "CORPUS_BUILD_FAILED"


# Bảng ánh xạ từ mã lỗi miền sang HTTP status code và message mặc định
ERROR_HTTP_STATUS: dict[ErrorCode, tuple[int, str]] = {
    ErrorCode.NOT_FOUND: (404, "data not found"),
    ErrorCode.AGENT_TIMEOUT: (504, "agent timed out"),
    ErrorCode.AGENT_FAILED: (500, "agent failed"),
    ErrorCode.RATE_LIMITED: (429, "Too many requests. Please slow down."),
    ErrorCode.REQUEST_TIMEOUT: (504, "Request processing timed out"),
    ErrorCode.PROMPT_INJECTION_DETECTED: (400, "Potential prompt injection detected. Request rejected."),
    ErrorCode.TOOL_BUDGET_EXCEEDED: (429, "Tool call budget exceeded for this request"),
    ErrorCode.PAYLOAD_TOO_LARGE: (413, "Payload too large. Maximum allowed size is 1GB."),
    ErrorCode.TASK_NOT_FOUND: (404, "Corpus task not found"),
    ErrorCode.CORPUS_BUILD_FAILED: (500, "Corpus build failed"),
}


class AppError(Exception):
    """Base cho mọi lỗi đã biết trước."""

    def __init__(self, code: ErrorCode, detail: str | None = None):
        status, default_detail = ERROR_HTTP_STATUS.get(code, (500, "internal error"))
        self.code = code
        self.status_code = status
        self.detail = detail or default_detail
        super().__init__(self.detail)
