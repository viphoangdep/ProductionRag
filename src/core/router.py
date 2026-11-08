"""Query Router & Intent Classification for dynamic agent routing."""
from __future__ import annotations

import logging
from typing import Any, Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import SystemMessage

from config.prompts import ROUTER_INTENT_PROMPT, SIMPLE_PROMPT, SYSTEM_PROMPT
from core.interfaces.router import BaseQueryRouter, QueryIntent, RouteDecision
from tools.search_tool import search_documents
from util.string_util import normalize_ws

log = logging.getLogger("ai-service")

# Semantic intent keywords/phrases for fallback when model is unavailable or in offline testing
_CHITCHAT_KEYWORDS = {
    "chao", "xin chao", "hello", "hi", "hey", "yo",
    "cam on", "thank", "thanks", "thank you",
    "tam biet", "bye", "goodbye",
    "who are you", "ban la ai", "what is your name", "my name is",
    "toi la", "minh la", "i am",
}


class QueryRouter(BaseQueryRouter):
    """Classifier and router that directs queries to weak (chitchat) or strong (RAG) models."""

    def __init__(self, weak_model: Any = None):
        self.weak_model = weak_model

    def _classify_lexical(self, query: str) -> QueryIntent:
        """Deterministic fallback based on semantic token taxonomy, without regex."""
        normalized = normalize_ws(query).lower().strip()
        # Remove common punctuation
        for ch in "?!.,:;-~()[]":
            normalized = normalized.replace(ch, " ")
        normalized = normalize_ws(normalized)

        if not normalized:
            return QueryIntent.CHITCHAT

        words = normalized.split()
        # Short conversational queries (< 50 chars and <= 5 words)
        if len(normalized) < 50 and len(words) <= 5:
            for kw in _CHITCHAT_KEYWORDS:
                if kw in normalized:
                    return QueryIntent.CHITCHAT

        return QueryIntent.RAG_SEARCH

    def classify_intent(self, query: str, **kwargs: Any) -> QueryIntent:
        """Classify user query into QueryIntent using LLM when available, falling back to lexical taxonomy."""
        model = kwargs.get("model") or self.weak_model
        clean_query = normalize_ws(query)

        if model is not None:
            prompt = ROUTER_INTENT_PROMPT.format(query=clean_query)
            try:
                res = model.invoke(prompt)
                content = getattr(res, "content", str(res)).strip().upper()
                if "CHITCHAT" in content:
                    return QueryIntent.CHITCHAT
                if "RAG_SEARCH" in content or "RAG" in content:
                    return QueryIntent.RAG_SEARCH
            except Exception as e:
                log.warning("LLM intent classification failed: %s; falling back to taxonomy", e)

        return self._classify_lexical(clean_query)

    def route(self, query: str, context: dict[str, Any] | None = None) -> RouteDecision:
        """Produce a complete RouteDecision for model tier, prompt, and tools."""
        ctx = context or {}
        model = ctx.get("weak_model") or self.weak_model
        intent = self.classify_intent(query, model=model)

        if intent == QueryIntent.CHITCHAT:
            return RouteDecision(
                intent=QueryIntent.CHITCHAT,
                model_tier="weak",
                tools_allowed=[],
                system_prompt=SIMPLE_PROMPT,
                reason="User query classified as casual chitchat or greeting",
            )

        return RouteDecision(
            intent=QueryIntent.RAG_SEARCH,
            model_tier="strong",
            tools_allowed=["search_documents"],
            system_prompt=SYSTEM_PROMPT,
            reason="User query requires domain documentation and search tool",
        )


_DEFAULT_ROUTER = QueryRouter()


def get_default_router() -> QueryRouter:
    return _DEFAULT_ROUTER


def _last_text(messages: list) -> str:
    if not messages:
        return ""
    c = getattr(messages[-1], "content", messages[-1])
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c)


def is_simple(query_norm: str, weak_model: Any = None) -> bool:
    """Check if query is simple chitchat using QueryRouter intent classification."""
    intent = get_default_router().classify_intent(query_norm, model=weak_model)
    return intent == QueryIntent.CHITCHAT


def pick_prompt(query_norm: str, weak_model: Any = None) -> str:
    """Pick system prompt according to query intent."""
    decision = get_default_router().route(query_norm, context={"weak_model": weak_model})
    return decision.system_prompt


@wrap_model_call
def dynamic_router(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    qn = normalize_ws(_last_text(request.messages))
    ctx = request.runtime.context or {}
    router = ctx.get("router") or get_default_router()
    decision = router.route(qn, context=ctx)

    if decision.intent == QueryIntent.CHITCHAT:
        log.info("route=CHITCHAT model=%s q=%.100s", ctx.get("weak_name", "?"), qn)
        return handler(
            request.override(
                model=ctx["weak_model"],
                tools=[],
                system_message=SystemMessage(content=decision.system_prompt),
            )
        )

    log.info("route=RAG_SEARCH model=%s q=%.100s", ctx.get("strong_name", "?"), qn)
    return handler(
        request.override(
            model=ctx["strong_model"],
            tools=[search_documents],
            system_message=SystemMessage(content=decision.system_prompt),
        )
    )
