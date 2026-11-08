"""Unit tests for multi-user, multi-session management and PostgresSessionStore."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from core.interfaces.session_store import BaseSessionStore
from core.session.session_manager import SessionManager
from infra.postgres_session_store import PostgresSessionStore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from model.req.chat_req import ChatQueryRequest
from model.res.chat_res import ChatQueryResponse
from service.chat_service import ChatService


def test_session_stores_inherit_interface():
    assert issubclass(SessionManager, BaseSessionStore)
    assert issubclass(PostgresSessionStore, BaseSessionStore)


def test_session_store_langchain_base_history():
    mgr = SessionManager()
    history = mgr.get_session_history("sess_1", user_id="user_test")
    assert isinstance(history, BaseChatMessageHistory)

    # Use standard LangChain methods directly on history
    history.add_user_message("Hello LangChain")
    history.add_ai_message("Hello Human")

    # Verify get_messages returns BaseMessage objects
    msgs = mgr.get_messages("sess_1", user_id="user_test")
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[0].content == "Hello LangChain"
    assert msgs[1].content == "Hello Human"

    # Verify backwards-compatible get_history returns dicts
    hist_dicts = mgr.get_history("sess_1", user_id="user_test")
    assert hist_dicts[0] == {"role": "user", "content": "Hello LangChain"}
    assert hist_dicts[1] == {"role": "assistant", "content": "Hello Human"}

    # Verify prompt formatting
    recent_text = mgr.get_recent_messages_text("sess_1", n=2, user_id="user_test")
    assert "User: Hello LangChain" in recent_text
    assert "Assistant: Hello Human" in recent_text


def test_multi_user_isolation_in_session_manager():
    mgr = SessionManager()

    # User A and User B share the exact same session_id string "sess_work"
    mgr.record_turn("sess_work", "User A question 1", "User A answer 1", user_id="user_a")
    mgr.record_turn("sess_work", "User B question 1", "User B answer 1", user_id="user_b")

    # Verify history isolation
    hist_a = mgr.get_history("sess_work", user_id="user_a")
    hist_b = mgr.get_history("sess_work", user_id="user_b")

    assert len(hist_a) == 2
    assert hist_a[0]["content"] == "User A question 1"
    assert hist_a[1]["content"] == "User A answer 1"

    assert len(hist_b) == 2
    assert hist_b[0]["content"] == "User B question 1"
    assert hist_b[1]["content"] == "User B answer 1"

    # Verify listing isolation
    sessions_a = mgr.list_user_sessions(user_id="user_a")
    sessions_b = mgr.list_user_sessions(user_id="user_b")

    assert len(sessions_a) == 1
    assert sessions_a[0]["session_id"] == "sess_work"
    assert sessions_a[0]["user_id"] == "user_a"

    assert len(sessions_b) == 1
    assert sessions_b[0]["session_id"] == "sess_work"
    assert sessions_b[0]["user_id"] == "user_b"

    # Reset User A's session -> User B remains untouched
    assert mgr.reset_session("sess_work", user_id="user_a") is True
    assert len(mgr.get_history("sess_work", user_id="user_a")) == 0
    assert len(mgr.get_history("sess_work", user_id="user_b")) == 2


def test_multi_session_per_user():
    mgr = SessionManager()

    mgr.record_turn("session_alpha", "Question 1", "Answer 1", user_id="user_bob")
    mgr.record_turn("session_beta", "Question 2", "Answer 2", user_id="user_bob")

    sessions = mgr.list_user_sessions(user_id="user_bob")
    assert len(sessions) == 2
    ids = {s["session_id"] for s in sessions}
    assert ids == {"session_alpha", "session_beta"}

    # Bulk delete for user
    deleted_count = mgr.delete_user_sessions(user_id="user_bob")
    assert deleted_count == 2
    assert len(mgr.list_user_sessions(user_id="user_bob")) == 0


def test_postgres_session_store_mocked_db():
    """Verify PostgresSessionStore operations using mocked psycopg connection and PostgresChatMessageHistory."""
    with patch("infra.postgres_session_store.psycopg.connect") as mock_connect, \
         patch("infra.postgres_session_store.PostgresChatMessageHistory") as mock_history_cls:

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        mock_history_instance = MagicMock()
        mock_history_cls.return_value = mock_history_instance

        store = PostgresSessionStore(
            connection_string="postgresql://test:test@localhost:5432/test_db",
            auto_create_tables=False,
        )

        assert isinstance(store, BaseSessionStore)

        # 1. Test record_turn
        store.record_turn(
            session_id="sess_100",
            user_query="How do I change ISO?",
            assistant_response="Press the ISO button.",
            user_id="alice",
        )

        mock_history_instance.add_user_message.assert_called_once_with("How do I change ISO?")
        mock_history_instance.add_ai_message.assert_called_once_with("Press the ISO button.")
        mock_cursor.execute.assert_called()  # user_sessions upsert

        # 2. Test reset_session
        mock_cursor.fetchone.return_value = (1,)  # ID returned
        deleted = store.reset_session("sess_100", user_id="alice")
        assert deleted is True
        mock_history_instance.clear.assert_called_once()

        # 3. Test list_user_sessions
        mock_cursor.fetchall.return_value = [
            ("sess_100", "alice", "Nikon Q&A", "Press the ISO button.", None, None)
        ]
        user_sessions = store.list_user_sessions(user_id="alice")
        assert len(user_sessions) == 1
        assert user_sessions[0]["session_id"] == "sess_100"
        assert user_sessions[0]["user_id"] == "alice"

        # 4. Test get_session_history returns LangChain PostgresChatMessageHistory
        history = store.get_session_history("sess_100", user_id="alice")
        assert history == mock_history_instance


def test_chat_service_multi_user_flow():
    class DummyAgent:
        def invoke(self, payload, **kwargs):
            return {"messages": [{"role": "assistant", "content": "mock answer"}]}

    session_mgr = SessionManager()
    service = ChatService(
        agent=DummyAgent(),
        agent_context={},
        weak_model=MagicMock(),
        session_manager=session_mgr,
        conversation_analyzer=MagicMock(),
        grounding_verifier=MagicMock(),
    )
    service.analyzer.analyze_and_rewrite.return_value = (False, "", "hello")
    service.grounding_verifier.verify.return_value = {
        "is_grounded": True,
        "grounding_score": 1.0,
        "citations": [],
        "cited_chunk_ids": [],
        "unsupported_claims": [],
    }

    req = ChatQueryRequest(query="Xin chao", user_id="user_carol", session_id="carol_session_1")
    res = service.query(req)

    assert res.user_id == "user_carol"
    assert res.session_id == "carol_session_1"
    assert res.answer == "mock answer"

    # Verify session recorded under user_carol
    sessions = service.list_user_sessions(user_id="user_carol")
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "carol_session_1"

    # Verify history
    hist = service.get_session_history(session_id="carol_session_1", user_id="user_carol")
    assert len(hist) == 2
    assert hist[0]["content"] == "Xin chao"
    assert hist[1]["content"] == "mock answer"

