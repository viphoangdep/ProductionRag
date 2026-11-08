"""PostgreSQL Session Store adapter implementing BaseSessionStore using langchain-postgres."""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
import psycopg
from psycopg import sql

from core.interfaces.session_store import BaseSessionStore

log = logging.getLogger("ai-service")


class PostgresSessionStore(BaseSessionStore):
    """Multi-user and multi-session store persisting to PostgreSQL via langchain-postgres."""

    def __init__(
        self,
        connection_string: str,
        table_name: str = "langchain_chat_history",
        metadata_table: str = "user_sessions",
        max_history_turns: int = 6,
        auto_create_tables: bool = True,
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self.metadata_table = metadata_table
        self.max_history_turns = max_history_turns
        self.checkpointer = MemorySaver()
        if auto_create_tables:
            self._init_db()

    def _get_connection(self) -> psycopg.Connection:
        """Create a new database connection."""
        return psycopg.connect(self.connection_string)

    def _init_db(self) -> None:
        """Ensure message history and user_sessions tables exist in PostgreSQL."""
        try:
            with self._get_connection() as conn:
                # 1. Initialize langchain-postgres message table
                PostgresChatMessageHistory.create_tables(conn, self.table_name)

                # 2. Initialize user session metadata table
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.metadata_table} (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(128) NOT NULL,
                            session_id VARCHAR(128) NOT NULL,
                            title VARCHAR(256),
                            last_message TEXT,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            CONSTRAINT uq_user_session UNIQUE (user_id, session_id)
                        );
                        CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON {self.metadata_table}(user_id);
                        CREATE INDEX IF NOT EXISTS idx_user_sessions_updated_at ON {self.metadata_table}(updated_at DESC);
                        """
                    )
                conn.commit()
                log.info("Postgres session store tables verified (history=%s, meta=%s)",
                         self.table_name, self.metadata_table)
        except Exception as e:
            log.warning("Could not initialize PostgreSQL session tables: %s (will retry on usage)", e)

    def _composite_key(self, session_id: str, user_id: str = "default_user") -> str:
        """Generate partition key for langchain_postgres."""
        return f"{user_id}:{session_id}" if user_id != "default_user" else session_id

    def get_session_history(
        self,
        session_id: str,
        user_id: str = "default_user",
    ) -> PostgresChatMessageHistory:
        """Retrieve the LangChain PostgresChatMessageHistory instance for the given session."""
        composite_key = self._composite_key(session_id, user_id)
        conn = self._get_connection()
        return PostgresChatMessageHistory(
            self.table_name,
            composite_key,
            sync_connection=conn,
        )

    def record_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str = "",
        assistant_answer: str = "",
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> None:
        """Record turn into PostgreSQL via langchain-postgres and update user_sessions metadata."""
        ans = assistant_response or assistant_answer
        composite_key = self._composite_key(session_id, user_id)

        with self._get_connection() as conn:
            # 1. Append messages via langchain-postgres
            history = PostgresChatMessageHistory(
                self.table_name,
                composite_key,
                sync_connection=conn,
            )
            if user_query:
                history.add_user_message(user_query)
            if ans:
                history.add_ai_message(ans)

            # 2. Update user_sessions metadata
            snippet = ans[:100] if ans else user_query[:100]
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.metadata_table} (user_id, session_id, last_message, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, session_id)
                    DO UPDATE SET last_message = EXCLUDED.last_message, updated_at = CURRENT_TIMESTAMP;
                    """,
                    (user_id, session_id, snippet),
                )
            conn.commit()

    def get_messages(
        self,
        session_id: str,
        user_id: str = "default_user",
    ) -> list[BaseMessage]:
        """Retrieve recent conversation BaseMessage instances from PostgreSQL."""
        composite_key = self._composite_key(session_id, user_id)
        with self._get_connection() as conn:
            history = PostgresChatMessageHistory(
                self.table_name,
                composite_key,
                sync_connection=conn,
            )
            messages = history.messages

        return messages[-self.max_history_turns :]

    def get_history(self, session_id: str, user_id: str = "default_user") -> list[dict[str, str]]:
        """Retrieve recent conversation turns as dicts from PostgreSQL."""
        messages = self.get_messages(session_id, user_id=user_id)
        turns = []
        for m in messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            turns.append({"role": role, "content": str(m.content)})
        return turns

    def get_recent_messages(self, session_id: str, n: int = 4, user_id: str = "default_user") -> list[dict[str, str]]:
        """Get the last N messages."""
        turns = self.get_history(session_id, user_id=user_id)
        return turns[-n:]

    def get_recent_messages_text(self, session_id: str, n: int = 4, user_id: str = "default_user") -> str:
        """Format recent N messages as a plain-text prompt string."""
        msgs = self.get_recent_messages(session_id, n=n, user_id=user_id)
        if not msgs:
            return ""
        lines = []
        for m in msgs:
            role = "User" if m.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {m.get('content', '')}")
        return "\n".join(lines)

    def reset_session(self, session_id: str, user_id: str = "default_user") -> bool:
        """Clear conversation history and delete session metadata."""
        composite_key = self._composite_key(session_id, user_id)
        with self._get_connection() as conn:
            history = PostgresChatMessageHistory(
                self.table_name,
                composite_key,
                sync_connection=conn,
            )
            history.clear()

            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.metadata_table} WHERE user_id = %s AND session_id = %s RETURNING id;",
                    (user_id, session_id),
                )
                deleted = cur.fetchone() is not None
            conn.commit()
            return deleted

    def list_user_sessions(self, user_id: str = "default_user") -> list[dict[str, Any]]:
        """List all active sessions for a user from user_sessions table."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT session_id, user_id, title, last_message, updated_at, created_at
                    FROM {self.metadata_table}
                    WHERE user_id = %s
                    ORDER BY updated_at DESC;
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "session_id": r[0],
                "user_id": r[1],
                "title": r[2],
                "last_message": r[3] or "",
                "updated_at": str(r[4]) if r[4] else "",
                "created_at": str(r[5]) if r[5] else "",
            })
        return results

    def delete_user_sessions(self, user_id: str = "default_user") -> int:
        """Delete all sessions belonging to a specific user."""
        sessions = self.list_user_sessions(user_id=user_id)
        count = 0
        for s in sessions:
            if self.reset_session(s["session_id"], user_id=user_id):
                count += 1
        return count
