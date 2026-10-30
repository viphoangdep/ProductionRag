"""Quản lý phiên hội thoại (multi-turn conversation) và checkpointer cho LangGraph."""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

log = logging.getLogger("ai-service")


class SessionManager:
    """Quản lý hội thoại đa lượt trong RAM kèm TTL và giới hạn dung lượng để tránh rò rỉ bộ nhớ."""

    def __init__(self, ttl_minutes: int = 60, max_sessions: int = 1000, max_history_turns: int = 6):
        self.ttl_seconds = ttl_minutes * 60
        self.max_sessions = max_sessions
        self.max_history_turns = max_history_turns
        self.checkpointer = MemorySaver()
        self._history: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._last_active: dict[str, float] = {}

    def _cleanup(self, now: float | None = None) -> None:
        """Dọn dẹp session đã hết hạn theo TTL hoặc vượt quá max_sessions."""
        if now is None:
            now = time.time()
        expired = [sid for sid, t in self._last_active.items() if now - t > self.ttl_seconds]
        for sid in expired:
            self.reset_session(sid)

        while len(self._history) > self.max_sessions:
            oldest_sid, _ = self._history.popitem(last=False)
            self._last_active.pop(oldest_sid, None)
            log.info("session evicted due to capacity: %s", oldest_sid)

    def touch(self, session_id: str) -> None:
        """Cập nhật thời điểm hoạt động gần nhất nếu session đã tồn tại."""
        self._cleanup()
        if session_id in self._history:
            self._last_active[session_id] = time.time()
            self._history.move_to_end(session_id)

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Lấy danh sách các lượt hội thoại gần nhất (User & Assistant)."""
        if session_id in self._history:
            self.touch(session_id)
        hist = self._history.get(session_id, [])
        return hist[-self.max_history_turns :]

    def get_recent_messages(self, session_id: str, n: int = 4) -> list[dict[str, str]]:
        """Lấy n message gần nhất (mặc định 4 message = 2 lượt gần nhất)."""
        if session_id in self._history:
            self.touch(session_id)
        hist = self._history.get(session_id, [])
        return hist[-n:]

    def get_recent_messages_text(self, session_id: str, n: int = 4) -> str:
        """Định dạng n message gần nhất thành chuỗi văn bản cho Prompt."""
        msgs = self.get_recent_messages(session_id, n=n)
        if not msgs:
            return ""
        lines = []
        for m in msgs:
            role = "User" if m.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {m.get('content', '')}")
        return "\n".join(lines)

    def get_history_text(self, session_id: str) -> str:
        """Định dạng lịch sử thành chuỗi văn bản cho Prompt contextualization."""
        turns = self.get_history(session_id)
        if not turns:
            return ""
        lines = []
        for t in turns:
            role = "User" if t.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {t.get('content', '')}")
        return "\n".join(lines)

    def record_turn(self, session_id: str, user_query: str, assistant_answer: str) -> None:
        """Ghi nhận lượt hội thoại mới."""
        if session_id not in self._history:
            self._history[session_id] = []
        self._last_active[session_id] = time.time()
        self._history.move_to_end(session_id)
        self._history[session_id].append({"role": "user", "content": user_query})
        self._history[session_id].append({"role": "assistant", "content": assistant_answer})

        # Giữ số lượt vừa phải trong RAM
        if len(self._history[session_id]) > self.max_history_turns * 2:
            self._history[session_id] = self._history[session_id][-self.max_history_turns * 2 :]

        self._cleanup()
        if len(self._history[session_id]) > self.max_history_turns * 2:
            self._history[session_id] = self._history[session_id][-self.max_history_turns * 2 :]

    def reset_session(self, session_id: str) -> bool:
        """Xóa toàn bộ lịch sử và checkpoint của session."""
        existed = session_id in self._history or session_id in self._last_active
        self._history.pop(session_id, None)
        self._last_active.pop(session_id, None)

        # Xóa checkpoint LangGraph nếu có
        try:
            storage = getattr(self.checkpointer, "storage", None)
            if storage is not None and session_id in storage:
                storage.pop(session_id, None)
        except Exception:
            pass

        log.info("session reset: %s (existed=%s)", session_id, existed)
        return existed
