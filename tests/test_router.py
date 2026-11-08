"""Unit tests for QueryRouter and BaseQueryRouter interface."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from core.interfaces.router import BaseQueryRouter, QueryIntent, RouteDecision
from core.router import QueryRouter, dynamic_router, is_simple, pick_prompt
from config.prompts import SIMPLE_PROMPT, SYSTEM_PROMPT


def test_base_query_router_is_abstract():
    with pytest.raises(TypeError):
        BaseQueryRouter()  # type: ignore


def test_query_router_inherits_interface():
    router = QueryRouter()
    assert isinstance(router, BaseQueryRouter)


def test_query_router_with_llm_chitchat():
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = "CHITCHAT"

    router = QueryRouter(weak_model=mock_model)
    intent = router.classify_intent("Hello assistant!")
    assert intent == QueryIntent.CHITCHAT
    mock_model.invoke.assert_called_once()


def test_query_router_with_llm_rag_search():
    mock_model = MagicMock()
    mock_model.invoke.return_value.content = "RAG_SEARCH"

    router = QueryRouter(weak_model=mock_model)
    intent = router.classify_intent("Pin của máy Nikon Zfc dùng loại nào?")
    assert intent == QueryIntent.RAG_SEARCH
    mock_model.invoke.assert_called_once()


def test_query_router_lexical_fallback():
    router = QueryRouter(weak_model=None)

    # Chitchat / Greetings
    assert router.classify_intent("xin chao") == QueryIntent.CHITCHAT
    assert router.classify_intent("Hello!") == QueryIntent.CHITCHAT
    assert router.classify_intent("Hi there") == QueryIntent.CHITCHAT
    assert router.classify_intent("cam on ban") == QueryIntent.CHITCHAT
    assert router.classify_intent("bye") == QueryIntent.CHITCHAT

    # Technical / RAG questions
    assert router.classify_intent("May anh nay co chong nuoc khong?") == QueryIntent.RAG_SEARCH
    assert router.classify_intent("Thong so ky thuat cua Nikon Zfc") == QueryIntent.RAG_SEARCH
    assert router.classify_intent("Cach cai dat menu chup dem") == QueryIntent.RAG_SEARCH


def test_route_decision():
    router = QueryRouter(weak_model=None)

    # Chitchat route
    decision_chitchat = router.route("Hello ban oi")
    assert decision_chitchat.intent == QueryIntent.CHITCHAT
    assert decision_chitchat.model_tier == "weak"
    assert decision_chitchat.tools_allowed == []
    assert decision_chitchat.system_prompt == SIMPLE_PROMPT

    # RAG route
    decision_rag = router.route("Pin Nikon Zfc dung duoc bao lau?")
    assert decision_rag.intent == QueryIntent.RAG_SEARCH
    assert decision_rag.model_tier == "strong"
    assert decision_rag.tools_allowed == ["search_documents"]
    assert decision_rag.system_prompt == SYSTEM_PROMPT


def test_is_simple_and_pick_prompt_compatibility():
    assert is_simple("Hello") is True
    assert pick_prompt("Hello") == SIMPLE_PROMPT

    assert is_simple("Huong dan su dung pin may anh Nikon Zfc") is False
    assert pick_prompt("Huong dan su dung pin may anh Nikon Zfc") == SYSTEM_PROMPT

