"""Quản lý phiên hội thoại đa người dùng (multi-user) và đa phiên (multi-session) trong RAM dựa trên LangChain InMemoryChatMessageHistory."""
from __future__ import annotations

import logging
import time
from collections import OrderedDict, defaultdict
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from core.interfaces.session_store import BaseSessionStore

log = logging.getLogger("ai-service")


class SessionManager(BaseSessionStore):
    """Quản lý hội thoại đa lượt và đa người dùng trong RAM dựa trên LangChain InMemoryChatMessageHistory kèm TTL."""

    def __init__(self, ttl_minutes: int = 60, max_sessions: int = 1000, max_history_turns: int = 6):
        self.ttl_seconds = ttl_minutes * 60
        self.max_sessions = max_sessions
        self.max_history_turns = max_history_turns
        self.checkpointer = MemorySaver()
        self._history: OrderedDict[str, InMemoryChatMessageHistory] = OrderedDict()
        self._last_active: dict[str, float] = {}
        self._user_sessions: dict[str, set[str]] = defaultdict(set)
        self._session_user: dict[str, str] = {}

    def _resolve_key(self, session_id: str, user_id: str = "default_user") -> str:
        """Resolve composite key or fallback to raw session_id for backwards compatibility."""
        composite = f"{user_id}:{session_id}"
        if composite in self._history:
            return composite
        if session_id in self._history:
            return session_id
        return composite if user_id != "default_user" else session_id

    def _cleanup(self, now: float | None = None) -> None:
        """Dọn dẹp session đã hết hạn theo TTL hoặc vượt quá max_sessions."""
        if now is None:
            now = time.time()
        expired = [sid for sid, t in self._last_active.items() if now - t > self.ttl_seconds]
        for key in expired:
            self._evict_key(key)

        while len(self._history) > self.max_sessions:
            oldest_key, hist = self._history.popitem(last=False)
            self._last_active.pop(oldest_key, None)
            u = self._session_user.pop(oldest_key, None)
            if u and u in self._user_sessions:
                self._user_sessions[u].discard(oldest_key.split(":")[-1])
            log.info("session evicted due to capacity: %s", oldest_key)

    def _evict_key(self, key: str) -> None:
        hist = self._history.pop(key, None)
        if hist is not None:
            hist.clear()
        self._last_active.pop(key, None)
        u = self._session_user.pop(key, None)
        sid = key.split(":")[-1]
        if u and u in self._user_sessions:
            self._user_sessions[u].discard(sid)

    def touch(self, key: str) -> None:
        """Cập nhật thời điểm hoạt động gần nhất."""
        self._cleanup()
        if key in self._history:
            self._last_active[key] = time.time()
            self._history.move_to_end(key)

    def get_session_history(
        self,
        session_id: str,
        user_id: str = "default_user",
    ) -> InMemoryChatMessageHistory:
        """Lấy hoặc khởi tạo instance InMemoryChatMessageHistory của LangChain cho session."""
        self._cleanup()
        key = self._resolve_key(session_id, user_id)
        if key not in self._history:
            self._history[key] = InMemoryChatMessageHistory()
            self._session_user[key] = user_id
            self._user_sessions[user_id].add(session_id)

        self.touch(key)
        return self._history[key]

    def get_messages(
        self,
        session_id: str,
        user_id: str = "default_user",
    ) -> list[BaseMessage]:
        """Lấy danh sách các BaseMessage (tối đa max_history_turns)."""
        key = self._resolve_key(session_id, user_id)
        if key not in self._history:
            return []
        self.touch(key)
        msgs = self._history[key].messages
        return msgs[-self.max_history_turns :]

    def get_history(self, session_id: str, user_id: str = "default_user") -> list[dict[str, str]]:
        """Lấy danh sách các lượt hội thoại gần nhất (User & Assistant) dạng dict."""
        messages = self.get_messages(session_id, user_id=user_id)
        turns = []
        for m in messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            turns.append({"role": role, "content": str(m.content)})
        return turns

    def get_recent_messages(self, session_id: str, n: int = 4, user_id: str = "default_user") -> list[dict[str, str]]:
        """Lấy n message gần nhất dạng dict."""
        key = self._resolve_key(session_id, user_id)
        if key not in self._history:
            return []
        self.touch(key)
        msgs = self._history[key].messages[-n:]
        turns = []
        for m in msgs:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            turns.append({"role": role, "content": str(m.content)})
        return turns

    def get_recent_messages_text(self, session_id: str, n: int = 4, user_id: str = "default_user") -> str:
        """Định dạng n message gần nhất thành chuỗi văn bản cho Prompt."""
        msgs = self.get_recent_messages(session_id, n=n, user_id=user_id)
        if not msgs:
            return ""
        lines = []
        for m in msgs:
            role = "User" if m.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {m.get('content', '')}")
        return "\n".join(lines)

    def get_history_text(self, session_id: str, user_id: str = "default_user") -> str:
        """Định dạng lịch sử thành chuỗi văn bản cho Prompt contextualization."""
        turns = self.get_history(session_id, user_id=user_id)
        if not turns:
            return ""
        lines = []
        for t in turns:
            role = "User" if t.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {t.get('content', '')}")
        return "\n".join(lines)

    def record_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str = "",
        assistant_answer: str = "",
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> None:
        """Ghi nhận lượt hội thoại mới cho user và session."""
        ans = assistant_response or assistant_answer
        history = self.get_session_history(session_id, user_id=user_id)
        if user_query:
            history.add_user_message(user_query)
        if ans:
            history.add_ai_message(ans)

        key = self._resolve_key(session_id, user_id)
        self.touch(key)

        # Giới hạn số lượng tin nhắn trong RAM
        if len(history.messages) > self.max_history_turns * 2:
            history.messages = history.messages[-self.max_history_turns * 2 :]

        self._cleanup()

    def reset_session(self, session_id: str, user_id: str = "default_user") -> bool:
        """Xóa toàn bộ lịch sử và checkpoint của session."""
        key = self._resolve_key(session_id, user_id)
        existed = key in self._history or key in self._last_active
        if key in self._history:
            self._history[key].clear()
        self._evict_key(key)

        # Xóa checkpoint LangGraph nếu có
        try:
            storage = getattr(self.checkpointer, "storage", None)
            if storage is not None:
                storage.pop(key, None)
                storage.pop(session_id, None)
        except Exception:
            pass
        return existed

    def list_user_sessions(self, user_id: str = "default_user") -> list[dict[str, Any]]:
        """Lấy danh sách tóm tắt toàn bộ session của một user."""
        self._cleanup()
        results = []
        session_ids = sorted(self._user_sessions.get(user_id, set()))
        for sid in session_ids:
            key = f"{user_id}:{sid}" if user_id != "default_user" else sid
            history = self._history.get(key)
            msgs = history.messages if history else []
            last_msg = str(msgs[-1].content) if msgs else ""
            results.append({
                "session_id": sid,
                "user_id": user_id,
                "message_count": len(msgs),
                "last_active": self._last_active.get(key, 0.0),
                "last_message": last_msg[:100],
            })
        return results

    def delete_user_sessions(self, user_id: str = "default_user") -> int:
        """Xóa toàn bộ session của một user."""
        session_ids = list(self._user_sessions.get(user_id, set()))
        count = 0
        for sid in session_ids:
            if self.reset_session(sid, user_id=user_id):
                count += 1
        self._user_sessions.pop(user_id, None)
        return count
