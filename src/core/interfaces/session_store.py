"""Abstract interface for multi-user and multi-session conversation storage aligned with LangChain."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class BaseSessionStore(ABC):
    """Interface for managing multi-user and multi-turn conversational history.

    Aligned with LangChain's BaseChatMessageHistory abstraction:
    Acts as a Session Provider / Factory producing `BaseChatMessageHistory` instances
    for individual sessions, while managing multi-user scoping, session lifecycle,
    and metadata persistence.
    """

    @abstractmethod
    def get_session_history(
        self,
        session_id: str,
        user_id: str = "default_user",
    ) -> BaseChatMessageHistory:
        """Retrieve the LangChain BaseChatMessageHistory instance for the given session."""
        raise NotImplementedError

    def get_messages(
        self,
        session_id: str,
        user_id: str = "default_user",
    ) -> list[BaseMessage]:
        """Get list of LangChain BaseMessage objects for a session."""
        history = self.get_session_history(session_id, user_id=user_id)
        return history.messages

    def record_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str = "",
        assistant_answer: str = "",
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> None:
        """Record a completed conversation turn into the user's session using BaseChatMessageHistory."""
        history = self.get_session_history(session_id, user_id=user_id)
        if user_query:
            history.add_user_message(user_query)
        ans = assistant_response or assistant_answer
        if ans:
            history.add_ai_message(ans)

    def get_history(
        self,
        session_id: str,
        user_id: str = "default_user",
    ) -> list[dict[str, str]]:
        """Get list of conversation turns as dicts for backwards-compatible API serialization."""
        messages = self.get_messages(session_id, user_id=user_id)
        turns = []
        for m in messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            turns.append({"role": role, "content": str(m.content)})
        return turns

    def get_recent_messages_text(
        self,
        session_id: str,
        n: int = 4,
        user_id: str = "default_user",
    ) -> str:
        """Get concatenated formatted text of recent N conversation messages for prompts."""
        messages = self.get_messages(session_id, user_id=user_id)[-n:]
        lines = []
        for m in messages:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    def reset_session(
        self,
        session_id: str,
        user_id: str = "default_user",
    ) -> bool:
        """Reset or clear conversation history for a given session. Returns True if existed."""
        history = self.get_session_history(session_id, user_id=user_id)
        if history.messages:
            history.clear()
            return True
        return False

    @abstractmethod
    def list_user_sessions(
        self,
        user_id: str = "default_user",
    ) -> list[dict[str, Any]]:
        """List active sessions belonging to a specific user with metadata."""
        raise NotImplementedError

    @abstractmethod
    def delete_user_sessions(
        self,
        user_id: str = "default_user",
    ) -> int:
        """Delete all sessions belonging to a user. Returns count of deleted sessions."""
        raise NotImplementedError
