from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1)
    user_id: str = Field(
        default="default_user",
        description="ID người dùng cho multi-user session management.",
    )
    session_id: str | None = Field(
        default=None,
        description="ID phiên hội thoại. Nếu không truyền, hệ thống tự sinh session mới.",
    )
    clear_history: bool = Field(
        default=False,
        description="Nếu True, xóa toàn bộ lịch sử trước đó của session_id này.",
    )
    role: str | None = None  # ponytail: chưa enforce, forward vào context chờ RBAC
