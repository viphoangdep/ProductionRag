"""Thẩm định tính Groundedness (chống ảo giác) và đối soát trích dẫn (citations)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from config.prompts import GROUNDING_VERIFY_PROMPT
from core.interfaces.verifier import BaseGroundingVerifier

log = logging.getLogger("ai-service")

# Regex bắt token trích dẫn dạng [chunk_id], [c1, c2], hoặc [id_123]
_CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9_\-,\s]+)\]")

# Mẫu câu từ chối an toàn khi tài liệu không có thông tin
_REFUSAL_PATTERN = re.compile(
    r"(không (có|chứa|tìm thấy)|chưa có|không đủ|does not contain|cannot find|not mentioned|"
    r"tài liệu (không|chưa)|không đề cập)\b",
    re.IGNORECASE,
)


def extract_cited_chunk_ids(answer: str, valid_chunk_ids: set[str] | None = None) -> list[str]:
    """Trích xuất danh sách chunk_id được trích dẫn trong văn bản."""
    found = set()
    for m in _CITATION_PATTERN.finditer(answer):
        raw = m.group(1).strip()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for p in parts:
            if valid_chunk_ids is None or p in valid_chunk_ids:
                found.add(p)
            elif not valid_chunk_ids and len(p) >= 2 and not p.isdigit():
                found.add(p)
    return sorted(found)


def is_grounded_refusal(answer: str) -> bool:
    """Kiểm tra xem câu trả lời có phải là từ chối hợp lệ do tài liệu không có dữ liệu."""
    a = answer.strip()
    return bool(_REFUSAL_PATTERN.search(a)) and len(a) < 300


def parse_grounding_json(raw_text: str) -> dict[str, Any] | None:
    """Trích xuất và parse JSON an toàn từ phản hồi của LLM."""
    text = raw_text.strip()
    # Tìm khối JSON ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # Tìm dấu ngoặc nhọn đầu tiên và cuối cùng
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


class GroundingVerifier(BaseGroundingVerifier):
    """Bộ thẩm định mức độ trung thực của câu trả lời so với tài liệu ngữ cảnh."""

    def __init__(self, weak_model=None, enable_eval: bool = True, threshold: float = 0.7):
        self.weak_model = weak_model
        self.enable_eval = enable_eval
        self.threshold = threshold

    def verify(
        self,
        answer: str,
        context_text: str,
        citations: list[dict],
        is_simple_query: bool = False,
    ) -> dict[str, Any]:
        """Thẩm định câu trả lời, đối soát trích dẫn và tính điểm tin cậy."""
        # 1. Tập hợp các chunk_id hợp lệ từ retrieval
        available_ids = {c["chunk_id"] for c in citations if "chunk_id" in c}
        cited_ids = extract_cited_chunk_ids(answer, available_ids if available_ids else None)

        # 2. Cập nhật cờ is_cited cho từng citation
        aligned_citations = []
        for c in citations:
            c_copy = dict(c)
            c_copy["is_cited"] = c_copy.get("chunk_id") in cited_ids
            aligned_citations.append(c_copy)

        # 3. Với câu hỏi chào hỏi/xã giao đơn giản
        if is_simple_query:
            return {
                "is_grounded": True,
                "grounding_score": 1.0,
                "cited_chunk_ids": [],
                "unsupported_claims": [],
                "citations": aligned_citations,
            }

        # 4. Với trường hợp model chủ động từ chối vì không có tài liệu
        if is_grounded_refusal(answer):
            return {
                "is_grounded": True,
                "grounding_score": 1.0,
                "cited_chunk_ids": cited_ids,
                "unsupported_claims": [],
                "citations": aligned_citations,
            }

        # 5. Nếu không có context tài liệu nhưng model trả lời khẳng định dài -> nghi ngờ hallucination
        if not context_text.strip() or not citations:
            return {
                "is_grounded": False,
                "grounding_score": 0.0,
                "cited_chunk_ids": [],
                "unsupported_claims": ["No context documents provided but the answer asserts facts"],
                "citations": aligned_citations,
            }

        # 6. LLM-as-a-judge (nếu bật và có weak_model)
        if self.enable_eval and self.weak_model is not None:
            try:
                # Cắt gọn context tối đa ~2500 ký tự để prompt gọn nhẹ
                ctx_snippet = context_text[:3000]
                prompt = GROUNDING_VERIFY_PROMPT.format(context=ctx_snippet, answer=answer)
                eval_res = self.weak_model.invoke(prompt)
                content = getattr(eval_res, "content", str(eval_res))
                parsed = parse_grounding_json(content)
                if parsed:
                    score = float(parsed.get("grounding_score", 1.0))
                    claims = parsed.get("unsupported_claims", [])
                    is_grounded = bool(parsed.get("is_grounded", score >= self.threshold))
                    return {
                        "is_grounded": is_grounded,
                        "grounding_score": score,
                        "cited_chunk_ids": cited_ids or parsed.get("cited_chunk_ids", []),
                        "unsupported_claims": claims,
                        "citations": aligned_citations,
                    }
            except Exception as e:
                log.warning("grounding eval via llm failed: %s, falling back to heuristics", e)

        # 7. Heuristic fallback khi không dùng LLM judge
        # Nếu câu trả lời có trích dẫn đúng ít nhất 1 chunk_id trong tài liệu
        if cited_ids:
            score = min(1.0, 0.7 + 0.1 * len(cited_ids))
            is_grounded = True
            unsupported = []
        else:
            # Không có trích dẫn nào trong câu trả lời
            score = 0.5
            is_grounded = score >= self.threshold
            unsupported = ["Answer lacks inline citation [chunk_id] supporting the claims"]

        return {
            "is_grounded": is_grounded,
            "grounding_score": score,
            "cited_chunk_ids": cited_ids,
            "unsupported_claims": unsupported,
            "citations": aligned_citations,
        }
