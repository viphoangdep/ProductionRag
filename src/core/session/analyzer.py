"""Bộ phân tích chủ đề hội thoại: kiểm tra liên quan, tóm tắt 4 message gần nhất và rewrite query."""
from __future__ import annotations

import logging
import re

from config.prompts import (
    REWRITE_QUERY_WITH_SUMMARY_PROMPT,
    SUMMARIZE_MESSAGES_PROMPT,
    TOPIC_RELATION_PROMPT,
)
from core.router import is_simple

log = logging.getLogger("ai-service")

# Regex nhận diện các từ chỉ quan hệ/tham chiếu ngữ cảnh trước
_COREFERENCE_PATTERN = re.compile(
    r"\b(nó|cái này|cái đó|chúng|tính năng này|chế độ này|thứ hai|thứ ba|ở đâu|"
    r"bao nhiêu|tại sao|vậy còn|thế còn|cho tôi biết thêm|so sánh|trên máy đó)\b",
    re.IGNORECASE,
)


class ConversationAnalyzer:
    """Phân tích xem câu hỏi người dùng có cần context của 4 câu hội thoại trước không."""

    def __init__(self, weak_model=None):
        self.weak_model = weak_model

    def needs_context(self, question: str, recent_messages_text: str) -> bool:
        """Đánh giá xem câu hỏi có CẦN context của 4 câu hội thoại trước để hiểu hay không.

        - True: Cần context -> CONTINUE (tiếp nối, cần tóm tắt và rewrite).
        - False: Không cần context -> NEW_TOPIC (độc lập, input vào hệ thống luôn).
        """
        if not recent_messages_text.strip():
            return False
        if is_simple(question):
            return False

        if self.weak_model is None:
            return bool(_COREFERENCE_PATTERN.search(question))

        prompt = TOPIC_RELATION_PROMPT.format(
            recent_messages=recent_messages_text, question=question
        )
        try:
            res = self.weak_model.invoke(prompt)
            content = getattr(res, "content", str(res)).strip().upper()
            if "CONTINUE" in content or "RELATED" in content:
                return True
            if "NEW_TOPIC" in content or "NEW" in content:
                return False
        except Exception as e:
            log.warning("context requirement check failed: %s, falling back to heuristic", e)

        # Fallback nếu model lỗi hoặc trả về định dạng lạ
        return bool(_COREFERENCE_PATTERN.search(question))

    def is_related(self, question: str, recent_messages_text: str) -> bool:
        """Alias cho needs_context."""
        return self.needs_context(question, recent_messages_text)

    def summarize_recent_messages(self, recent_messages_text: str) -> str:
        """Tóm tắt 4 message gần nhất (ngữ cảnh, thiết bị, thông số đã nói)."""
        if not recent_messages_text.strip():
            return ""
        if self.weak_model is None:
            return recent_messages_text[:300]

        prompt = SUMMARIZE_MESSAGES_PROMPT.format(recent_messages=recent_messages_text)
        try:
            res = self.weak_model.invoke(prompt)
            summary = getattr(res, "content", str(res)).strip()
            return summary or recent_messages_text[:300]
        except Exception as e:
            log.warning("summarize recent messages failed: %s", e)
            return recent_messages_text[:300]

    def rewrite_query_with_summary(self, question: str, summary: str) -> str:
        """Nhét bản tóm tắt vào prompt và viết lại câu hỏi thành standalone search query."""
        if not summary.strip() or self.weak_model is None:
            return question

        prompt = REWRITE_QUERY_WITH_SUMMARY_PROMPT.format(summary=summary, question=question)
        try:
            res = self.weak_model.invoke(prompt)
            raw = getattr(res, "content", str(res)).strip()
            # Lấy dòng đầu tiên nếu model trả về nhiều dòng
            rewritten = raw.splitlines()[0].strip() if raw else question
            if rewritten and len(rewritten) >= 3:
                return rewritten
        except Exception as e:
            log.warning("rewrite query with summary failed: %s", e)

        return question

    def analyze_and_rewrite(
        self, question: str, recent_messages_text: str
    ) -> tuple[bool, str | None, str]:
        """
        Quy trình:
        1. Đánh giá xem câu hỏi có CẦN context của 4 câu hội thoại trước không.
        2. Nếu CÓ CẦN (is_continue=True):
           - Tóm tắt 4 câu hội thoại trước.
           - Nhét bản tóm tắt vào prompt, rewrite câu hỏi thành standalone query.
        3. Nếu KHÔNG CẦN (is_continue=False / NEW_TOPIC):
           - Coi là new topic, input câu hỏi gốc vào hệ thống luôn.

        Returns:
            is_continue (bool): True nếu cần context (tiếp nối), False nếu không cần (new topic).
            summary (str | None): Tóm tắt 4 message nếu cần context, None nếu new topic.
            final_query (str): Query sau khi rewrite nếu cần context, hoặc query gốc nếu new topic.
        """
        if not recent_messages_text.strip():
            return False, None, question

        needed = self.needs_context(question, recent_messages_text)
        if not needed:
            return False, None, question

        summary = self.summarize_recent_messages(recent_messages_text)
        rewritten = self.rewrite_query_with_summary(question, summary)
        return True, summary, rewritten

        summary = self.summarize_recent_messages(recent_messages_text)
        rewritten = self.rewrite_query_with_summary(question, summary)
        return True, summary, rewritten

