"""Lỗi nghiệp vụ: 1 class base + enum mã lỗi. Web layer map sang HTTP."""
from enum import Enum


class ErrorCode(Enum):
    """(HTTP status, message mặc định). Mỗi loại lỗi là 1 giá trị enum, không phải 1 class."""

    NOT_FOUND = (404, "data not found")
    AGENT_TIMEOUT = (504, "agent timed out")
    AGENT_FAILED = (500, "agent failed")
    RATE_LIMITED = (429, "Too many requests. Please slow down.")
    REQUEST_TIMEOUT = (504, "Request processing timed out")
    PROMPT_INJECTION_DETECTED = (400, "Potential prompt injection detected. Request rejected.")
    TOOL_BUDGET_EXCEEDED = (429, "Tool call budget exceeded for this request")


class AppError(Exception):
    """Base cho mọi lỗi đã biết trước. Ai cần hành vi riêng thì kế thừa class này."""

    def __init__(self, code: ErrorCode, detail: str | None = None):
        status, default_detail = code.value
        self.code = code
        self.status_code = status
        self.detail = detail or default_detail
        super().__init__(self.detail)
