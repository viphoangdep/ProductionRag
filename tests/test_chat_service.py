"""ChatService: lấy answer + citations từ kết quả agent, lỗi -> AppError + ErrorCode."""
from types import SimpleNamespace

import pytest
from langgraph.errors import GraphRecursionError

from core.exception.errors import AppError, ErrorCode
from model.req.chat_req import ChatQueryRequest
from service.chat_service import ChatService


class FakeAgent:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.context = None

    def invoke(self, payload, context=None):
        self.context = context
        if self.exc:
            raise self.exc
        return self.result


def _msg(content, name=None, artifact=None):
    return SimpleNamespace(content=content, name=name, artifact=artifact)


class FakeModel:
    def __init__(self, content="p1\np2"):
        self.content = content

    def invoke(self, prompt):
        return SimpleNamespace(content=self.content)


def _service(agent):
    return ChatService(agent, {"cfg": {"top_n": 5}}, weak_model=FakeModel(), n_paraphrases=3)


def test_answer_and_citations_from_agent_result():
    msgs = [_msg("tool call", name="search_documents",
                 artifact={"paraphrases": ["p1"],
                           "citations": [{"chunk_id": "c1", "rerank_score": 0.9}]}),
            _msg("final answer")]
    res = _service(FakeAgent(result={"messages": msgs})).query(
        ChatQueryRequest(query="  hoi   gi  "))
    assert res.answer == "final answer"
    assert res.query_norm == "hoi gi"
    assert res.paraphrases == ["p1"]
    assert res.citations[0].chunk_id == "c1"


def test_answer_from_content_blocks():
    msgs = [_msg([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])]
    res = _service(FakeAgent(result={"messages": msgs})).query(ChatQueryRequest(query="q"))
    assert res.answer == "a b"


def test_no_tool_call_gives_empty_citations():
    res = _service(FakeAgent(result={"messages": [_msg("plain")]})).query(
        ChatQueryRequest(query="q"))
    assert res.citations == []
    assert res.paraphrases == []


def test_recursion_error_becomes_agent_timeout():
    with pytest.raises(AppError) as ei:
        _service(FakeAgent(exc=GraphRecursionError("boom"))).query(ChatQueryRequest(query="q"))
    assert ei.value.code is ErrorCode.AGENT_TIMEOUT
    assert ei.value.status_code == 504


def test_agent_exception_becomes_agent_failed():
    with pytest.raises(AppError) as ei:
        _service(FakeAgent(exc=RuntimeError("ollama down"))).query(ChatQueryRequest(query="q"))
    assert ei.value.code is ErrorCode.AGENT_FAILED
    assert ei.value.status_code == 500


def test_search_cache_is_fresh_per_request():
    agent = FakeAgent(result={"messages": [_msg("ok")]})
    svc = _service(agent)
    svc.query(ChatQueryRequest(query="q"))
    assert agent.context["search_cache"] == {}
    assert agent.context["role"] is None


def test_response_contains_grounding_and_session_fields():
    msgs = [_msg("tool call", name="search_documents",
                 artifact={"paraphrases": ["p1"],
                           "citations": [{"chunk_id": "c1", "rerank_score": 0.9}]}),
            _msg("Máy ảnh Nikon Zfc có cổng sạc Type-C [c1].")]
    res = _service(FakeAgent(result={"messages": msgs})).query(
        ChatQueryRequest(query="Nikon Zfc dùng cổng gì?", session_id="test-sess-001"))
    assert res.session_id == "test-sess-001"
    assert res.is_grounded is True
    assert "c1" in res.cited_chunk_ids
    assert res.citations[0].is_cited is True


def test_reset_session_clears_history():
    svc = _service(FakeAgent(result={"messages": [_msg("ok")]}))
    svc.query(ChatQueryRequest(query="q1", session_id="s1"))
    assert len(svc.session_manager.get_history("s1")) == 2
    assert svc.reset_session("s1") is True
    assert len(svc.session_manager.get_history("s1")) == 0
