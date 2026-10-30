"""Unit tests cho tính năng hội thoại liên tục, phân tích quan hệ chủ đề và rewrite query."""
from types import SimpleNamespace

import pytest

from core.session.analyzer import ConversationAnalyzer
from core.session.session_manager import SessionManager
from model.req.chat_req import ChatQueryRequest
from service.chat_service import ChatService


class FakeAgent:
    def __init__(self, result=None):
        self.result = result
        self.invocations = []

    def invoke(self, payload, context=None, config=None):
        self.invocations.append({"payload": payload, "context": context, "config": config})
        return self.result


def _msg(content, name=None, artifact=None):
    return SimpleNamespace(content=content, name=name, artifact=artifact)


class SmartMockModel:
    def __init__(self, is_related=True, summary="Máy ảnh Nikon Zfc dùng pin EN-EL25", rewritten="Pin máy ảnh Nikon Zfc dùng được bao lâu?"):
        self.is_related = is_related
        self.summary = summary
        self.rewritten = rewritten

    def invoke(self, prompt):
        p = str(prompt)
        if "CONTINUE hoặc NEW_TOPIC" in p or "RELATED hoặc NEW_TOPIC" in p or "new_topic" in p.lower():
            return SimpleNamespace(content="CONTINUE" if self.is_related else "NEW_TOPIC")
        if "tóm tắt ngắn gọn" in p.lower() or "summarize" in p.lower():
            return SimpleNamespace(content=self.summary)
        if "standalone search query" in p.lower() or "viết lại câu hỏi" in p.lower():
            return SimpleNamespace(content=self.rewritten)
        return SimpleNamespace(content=self.rewritten)


def test_session_manager_record_and_get_history():
    sm = SessionManager(ttl_minutes=10, max_history_turns=3)
    sm.record_turn("s1", "Máy Zfc có cổng sạc gì?", "Có cổng USB-C.")
    hist = sm.get_history("s1")
    assert len(hist) == 2
    assert hist[0] == {"role": "user", "content": "Máy Zfc có cổng sạc gì?"}
    assert hist[1] == {"role": "assistant", "content": "Có cổng USB-C."}

    text = sm.get_history_text("s1")
    assert "User: Máy Zfc có cổng sạc gì?" in text
    assert "Assistant: Có cổng USB-C." in text


def test_session_manager_get_recent_messages_4():
    sm = SessionManager()
    for i in range(1, 5):
        sm.record_turn("s1", f"q{i}", f"a{i}")
    # Tổng cộng có 8 messages (4 lượt)
    all_msgs = sm.get_history("s1")
    assert len(all_msgs) == 6  # max_history_turns mặc định = 6

    # get_recent_messages(n=4) chỉ lấy 4 message gần nhất (2 lượt cuối)
    recent = sm.get_recent_messages("s1", n=4)
    assert len(recent) == 4
    assert recent[0]["content"] == "q3"
    assert recent[1]["content"] == "a3"
    assert recent[2]["content"] == "q4"
    assert recent[3]["content"] == "a4"

    recent_text = sm.get_recent_messages_text("s1", n=4)
    assert "q3" in recent_text
    assert "a4" in recent_text
    assert "q1" not in recent_text


def test_session_manager_reset():
    sm = SessionManager()
    sm.record_turn("s1", "q1", "a1")
    assert sm.reset_session("s1") is True
    assert sm.get_history("s1") == []
    assert sm.reset_session("s1") is False


def test_session_manager_ttl_expiry():
    sm = SessionManager(ttl_minutes=1)
    sm.record_turn("s1", "q1", "a1")
    # Giả lập thời gian trôi qua 120s
    sm._cleanup(now=sm._last_active["s1"] + 120)
    assert sm.get_history("s1") == []


def test_session_manager_capacity_eviction():
    sm = SessionManager(max_sessions=2)
    sm.record_turn("s1", "q1", "a1")
    sm.record_turn("s2", "q2", "a2")
    sm.record_turn("s3", "q3", "a3")
    # s1 phải bị evict vì cũ nhất
    assert "s1" not in sm._history
    assert "s2" in sm._history
    assert "s3" in sm._history


def test_conversation_analyzer_related_flow():
    mock_model = SmartMockModel(
        is_related=True,
        summary="Đang thảo luận về pin máy ảnh Nikon Zfc",
        rewritten="Thời lượng pin của máy ảnh Nikon Zfc là bao lâu?",
    )
    analyzer = ConversationAnalyzer(weak_model=mock_model)
    recent_text = "User: Nikon Zfc dùng pin gì?\nAssistant: Dùng pin EN-EL25 [c1]."
    is_related, summary, rewritten = analyzer.analyze_and_rewrite(
        "Pin của nó dùng được bao lâu?", recent_text
    )

    assert is_related is True
    assert summary == "Đang thảo luận về pin máy ảnh Nikon Zfc"
    assert rewritten == "Thời lượng pin của máy ảnh Nikon Zfc là bao lâu?"


def test_conversation_analyzer_new_topic_flow():
    mock_model = SmartMockModel(is_related=False)
    analyzer = ConversationAnalyzer(weak_model=mock_model)
    recent_text = "User: Nikon Zfc dùng pin gì?\nAssistant: Dùng pin EN-EL25 [c1]."
    is_related, summary, final_query = analyzer.analyze_and_rewrite(
        "Máy bay Boeing 747 bay được bao xa?", recent_text
    )

    assert is_related is False
    assert summary is None
    assert final_query == "Máy bay Boeing 747 bay được bao xa?"


def test_chat_service_generates_session_id_when_none():
    agent = FakeAgent(result={"messages": [_msg("answer")]})
    svc = ChatService(agent, {"cfg": {}}, weak_model=SimpleNamespace(invoke=lambda p: SimpleNamespace(content="")))
    res = svc.query(ChatQueryRequest(query="test"))
    assert res.session_id is not None
    assert len(res.session_id) > 0


def test_chat_service_reuses_provided_session_id():
    agent = FakeAgent(result={"messages": [_msg("answer")]})
    svc = ChatService(agent, {"cfg": {}}, weak_model=SimpleNamespace(invoke=lambda p: SimpleNamespace(content="")))
    res = svc.query(ChatQueryRequest(query="test", session_id="my-custom-session"))
    assert res.session_id == "my-custom-session"


def test_chat_service_related_topic_summarizes_and_rewrites():
    agent = FakeAgent(result={"messages": [_msg("Nikon Zfc dùng pin EN-EL25 [c1].")]})
    model = SmartMockModel(
        is_related=True,
        summary="Máy ảnh Nikon Zfc sử dụng pin EN-EL25.",
        rewritten="Pin máy ảnh Nikon Zfc dùng được bao lâu?",
    )
    svc = ChatService(agent, {"cfg": {}}, weak_model=model)

    # Turn 1
    svc.query(ChatQueryRequest(query="Máy ảnh Nikon Zfc dùng pin gì?", session_id="sess-1"))

    # Turn 2: hỏi tiếp phụ thuộc ngữ cảnh "của nó"
    res2 = svc.query(ChatQueryRequest(query="Pin của nó dùng được bao lâu?", session_id="sess-1"))

    assert res2.session_id == "sess-1"
    assert res2.is_new_topic is False
    assert res2.standalone_query == "Pin máy ảnh Nikon Zfc dùng được bao lâu?"

    # Agent phải nhận input đã được rewrite
    last_invocation = agent.invocations[-1]
    assert last_invocation["payload"]["messages"][0]["content"] == "Pin máy ảnh Nikon Zfc dùng được bao lâu?"


def test_chat_service_new_topic_passes_query_directly():
    agent = FakeAgent(result={"messages": [_msg("answer")]})
    model = SmartMockModel(is_related=False)
    svc = ChatService(agent, {"cfg": {}}, weak_model=model)

    # Turn 1
    svc.query(ChatQueryRequest(query="Máy ảnh Nikon Zfc dùng pin gì?", session_id="sess-2"))

    # Turn 2: hỏi chủ đề mới hoàn toàn
    res2 = svc.query(ChatQueryRequest(query="Thời tiết Hà Nội hôm nay thế nào?", session_id="sess-2"))

    assert res2.session_id == "sess-2"
    assert res2.is_new_topic is True
    assert res2.standalone_query is None

    # Agent phải nhận đúng query gốc, KHÔNG bị rewrite sang Nikon Zfc
    last_invocation = agent.invocations[-1]
    assert last_invocation["payload"]["messages"][0]["content"] == "Thời tiết Hà Nội hôm nay thế nào?"


def test_chat_service_clear_history_flag():
    agent = FakeAgent(result={"messages": [_msg("answer")]})
    svc = ChatService(agent, {"cfg": {}}, weak_model=SimpleNamespace(invoke=lambda p: SimpleNamespace(content="")))
    svc.query(ChatQueryRequest(query="q1", session_id="s_clear"))
    assert len(svc.session_manager.get_history("s_clear")) == 2

    # Lượt tiếp theo gửi clear_history=True
    svc.query(ChatQueryRequest(query="q2", session_id="s_clear", clear_history=True))
    # Lịch sử chỉ còn turn mới q2
    hist = svc.session_manager.get_history("s_clear")
    assert len(hist) == 2
    assert hist[0]["content"] == "q2"
