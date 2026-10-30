"""Chat use case: chạy agent -> câu trả lời + citations. HTTP layer chỉ gọi vào đây."""
from __future__ import annotations

import inspect
import logging
import time
import uuid

from langgraph.errors import GraphRecursionError

from config.prompts import HYDE_PROMPT, QUERY_CONTEXTUALIZE_PROMPT
from config.settings import settings
from core.exception.errors import AppError, ErrorCode
from core.grounding.verifier import GroundingVerifier
from core.router import is_simple
from core.security import ToolBudget, check_prompt_injection
from core.session.analyzer import ConversationAnalyzer
from core.session.session_manager import SessionManager
from model.req.chat_req import ChatQueryRequest
from model.res.chat_res import ChatQueryResponse
from util.string_util import dedupe_paraphrases, normalize_ws

log = logging.getLogger("ai-service")


def _text(msg) -> str:
    """Content của AI message: string hoặc list content-block -> text."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _artifact(msgs: list) -> dict:
    """Artifact của tool search gần nhất (citations + paraphrases)."""
    for m in reversed(msgs):
        if getattr(m, "name", "") == "search_documents" and getattr(m, "artifact", None):
            return m.artifact
    return {}


def _tool_context(msgs: list) -> str:
    """Nội dung context văn bản mà tool search_documents trả về."""
    for m in reversed(msgs):
        if getattr(m, "name", "") == "search_documents":
            return _text(m)
    return ""


class ChatService:
    def __init__(
        self,
        agent,
        agent_context: dict,
        weak_model,
        session_manager: SessionManager | None = None,
        grounding_verifier: GroundingVerifier | None = None,
        conversation_analyzer: ConversationAnalyzer | None = None,
        n_paraphrases: int = 3,
    ):
        self.agent = agent
        self.ctx = agent_context
        self.weak_model = weak_model  # tier 1, container đã dựng sẵn
        self.session_manager = session_manager or SessionManager()
        self.grounding_verifier = grounding_verifier or GroundingVerifier(weak_model=weak_model)
        self.analyzer = conversation_analyzer or ConversationAnalyzer(weak_model=weak_model)
        self.n_paraphrases = n_paraphrases

    def paraphrase(self, query_norm: str) -> list[str]:
        """HyDE: viết lại câu hỏi thành N query khác nhau để đi tìm tài liệu."""
        prompt = HYDE_PROMPT.format(n=self.n_paraphrases, question=query_norm)
        raw = self.weak_model.invoke(prompt).content
        out = dedupe_paraphrases(raw.splitlines(), query_norm, self.n_paraphrases)
        log.info("hyde n=%d", len(out))
        return out

    def contextualize_query(self, session_id: str, query_norm: str) -> str:
        """Dựa trên lịch sử hội thoại, viết lại câu hỏi thành standalone query nếu cần."""
        if not session_id:
            return query_norm
        recent_4_text = self.session_manager.get_recent_messages_text(session_id, n=4)
        _, _, final_query = self.analyzer.analyze_and_rewrite(query_norm, recent_4_text)
        return final_query

    def reset_session(self, session_id: str) -> bool:
        """Xóa lịch sử hội thoại của một session."""
        return self.session_manager.reset_session(session_id)

    def _invoke_agent(self, payload: dict, ctx: dict, config: dict | None):
        """Invoke agent tương thích cả LangGraph lẫn mock object trong tests."""
        sig = inspect.signature(self.agent.invoke)
        kwargs = {"context": ctx}
        if "config" in sig.parameters and config is not None:
            kwargs["config"] = config
        return self.agent.invoke(payload, **kwargs)

    def query(self, body: ChatQueryRequest):
        t0 = time.perf_counter()
        session_id = body.session_id or str(uuid.uuid4())
        if body.clear_history:
            self.reset_session(session_id)

        query_norm = normalize_ws(body.query)

        # Prompt injection guard (English injection / jailbreak patterns)
        if getattr(settings, "enable_injection_guard", True):
            check_prompt_injection(query_norm)

        # 1. Lấy 4 message gần nhất (2 lượt gần nhất)
        recent_4_text = self.session_manager.get_recent_messages_text(session_id, n=4)

        # 2. Bộ phân tích: xem câu query có CẦN context của 4 câu hội thoại trước không?
        # - Nếu CẦN (continue): tóm tắt 4 câu hội thoại trước, nhét vào prompt, rewrite query.
        # - Nếu KHÔNG CẦN (new topic): input trực tiếp query gốc vào hệ thống.
        is_continue, context_summary, rewritten_query = self.analyzer.analyze_and_rewrite(
            query_norm, recent_4_text
        )

        if is_continue:
            standalone_query = rewritten_query
            input_query = standalone_query
            log.info("topic=CONTINUE session=%s summary='%.60s' rewrite='%.60s' -> '%.60s'",
                     session_id, context_summary, query_norm, standalone_query)
        else:
            standalone_query = None
            input_query = body.query
            log.info("topic=NEW_TOPIC session=%s q=%.60s -> direct input",
                     session_id, query_norm)

        ctx = {
            **self.ctx,
            "role": body.role,
            "search_cache": {},
            "paraphrase": self.paraphrase,
            "session_id": session_id,
            "standalone_query": standalone_query,
            "tool_budget": ToolBudget(max_calls=getattr(settings, "tool_call_budget", 3)),
        }
        ctx.setdefault("cfg", {})

        agent_config = {"configurable": {"thread_id": session_id}}

        try:
            result = self._invoke_agent(
                {"messages": [{"role": "user", "content": input_query}]},
                ctx=ctx,
                config=agent_config,
            )
        except GraphRecursionError:
            log.warning("chat recursion-limit session=%s role=%s q=%.100s",
                        session_id, body.role, query_norm)
            raise AppError(ErrorCode.AGENT_TIMEOUT,
                           "Search took too many rounds, please ask a more specific question")
        except Exception:
            log.exception("chat agent failed session=%s role=%s q=%.100s",
                          session_id, body.role, query_norm)
            raise AppError(ErrorCode.AGENT_FAILED, "Agent failed to answer, please try again")

        msgs = result["messages"]
        raw_answer = _text(msgs[-1])
        art = _artifact(msgs)
        raw_citations = art.get("citations", [])
        tool_ctx = _tool_context(msgs)

        # Thẩm định Grounding và kiểm tra trích dẫn
        # Nếu là simple query hoặc agent không gọi search tool (không có citations và không có context),
        # thì đây là câu hỏi trò chuyện/phi RAG -> coi như is_simple_query để không phạt unsupported_claims
        has_rag_evidence = bool(tool_ctx.strip() or raw_citations)
        is_non_rag = is_simple(query_norm) or not has_rag_evidence

        grounding_res = self.grounding_verifier.verify(
            answer=raw_answer,
            context_text=tool_ctx,
            citations=raw_citations,
            is_simple_query=is_non_rag,
        )

        # Lưu lượt hội thoại vào SessionManager
        self.session_manager.record_turn(session_id, body.query, raw_answer)

        log.info(
            "chat out session=%s %.0fms citations=%d grounded=%s score=%s",
            session_id,
            (time.perf_counter() - t0) * 1000,
            len(grounding_res["citations"]),
            grounding_res["is_grounded"],
            grounding_res["grounding_score"],
        )

        return ChatQueryResponse(
            session_id=session_id,
            answer=raw_answer,
            query_norm=query_norm,
            standalone_query=standalone_query,
            is_new_topic=not is_continue,
            is_grounded=grounding_res["is_grounded"],
            grounding_score=grounding_res["grounding_score"],
            cited_chunk_ids=grounding_res["cited_chunk_ids"],
            unsupported_claims=grounding_res["unsupported_claims"],
            paraphrases=art.get("paraphrases", []),
            citations=grounding_res["citations"],
        )
