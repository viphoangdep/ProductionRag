"""Unit tests cho module Grounding: trích xuất citation, kiểm tra bám sát tài liệu & chống hallucination."""
from types import SimpleNamespace

import pytest

from core.grounding.verifier import (
    GroundingVerifier,
    extract_cited_chunk_ids,
    is_grounded_refusal,
    parse_grounding_json,
)


def test_extract_cited_chunk_ids():
    text = "Theo hướng dẫn [chunk_01], máy có 4 cổng kết nối [chunk_02, chunk_03]."
    cited = extract_cited_chunk_ids(text, {"chunk_01", "chunk_02", "chunk_03", "chunk_04"})
    assert cited == ["chunk_01", "chunk_02", "chunk_03"]


def test_extract_cited_chunk_ids_filters_invalid():
    text = "Xem tài liệu [valid_id] và [invalid_id]."
    cited = extract_cited_chunk_ids(text, {"valid_id"})
    assert cited == ["valid_id"]


def test_is_grounded_refusal():
    assert is_grounded_refusal("Tài liệu hướng dẫn được cung cấp không có thông tin về tính năng này.")
    assert is_grounded_refusal("Tôi không tìm thấy thông tin liên quan trong sổ tay hướng dẫn.")
    assert is_grounded_refusal("The provided manual does not contain information about Boeing 747.")
    assert not is_grounded_refusal("Máy ảnh Nikon Zfc có độ phân giải 20.9 MP và cổng USB Type-C.")


def test_parse_grounding_json_markdown_block():
    raw = """Dưới đây là kết quả đánh giá:
```json
{
  "is_grounded": true,
  "grounding_score": 0.95,
  "unsupported_claims": [],
  "cited_chunk_ids": ["c1"]
}
```
Hết."""
    parsed = parse_grounding_json(raw)
    assert parsed is not None
    assert parsed["is_grounded"] is True
    assert parsed["grounding_score"] == 0.95
    assert parsed["cited_chunk_ids"] == ["c1"]


def test_parse_grounding_json_raw_object():
    raw = '{"is_grounded": false, "grounding_score": 0.3, "unsupported_claims": ["claim 1"]}'
    parsed = parse_grounding_json(raw)
    assert parsed is not None
    assert parsed["is_grounded"] is False
    assert parsed["grounding_score"] == 0.3


def test_grounding_verifier_simple_query():
    verifier = GroundingVerifier(enable_eval=False)
    res = verifier.verify(
        answer="Xin chào! Tôi có thể giúp gì cho bạn?",
        context_text="",
        citations=[],
        is_simple_query=True,
    )
    assert res["is_grounded"] is True
    assert res["grounding_score"] == 1.0


def test_grounding_verifier_safe_refusal():
    verifier = GroundingVerifier(enable_eval=False)
    res = verifier.verify(
        answer="Tài liệu không có thông tin về giá bán.",
        context_text="Nikon Zfc manual...",
        citations=[{"chunk_id": "c1", "rerank_score": 0.1}],
    )
    assert res["is_grounded"] is True
    assert res["grounding_score"] == 1.0


def test_grounding_verifier_missing_context_unsupported():
    verifier = GroundingVerifier(enable_eval=False)
    res = verifier.verify(
        answer="Máy ảnh này có giá 20 triệu đồng.",
        context_text="",
        citations=[],
        is_simple_query=False,
    )
    assert res["is_grounded"] is False
    assert res["grounding_score"] == 0.0
    assert len(res["unsupported_claims"]) > 0


def test_grounding_verifier_citation_alignment():
    verifier = GroundingVerifier(enable_eval=False)
    citations = [
        {"chunk_id": "c1", "rerank_score": 0.9},
        {"chunk_id": "c2", "rerank_score": 0.8},
    ]
    res = verifier.verify(
        answer="Tính năng này được mô tả tại [c1].",
        context_text="[c1] Body text 1\n[c2] Body text 2",
        citations=citations,
    )
    assert res["is_grounded"] is True
    assert res["cited_chunk_ids"] == ["c1"]
    assert res["citations"][0]["chunk_id"] == "c1"
    assert res["citations"][0]["is_cited"] is True
    assert res["citations"][1]["chunk_id"] == "c2"
    assert res["citations"][1]["is_cited"] is False


def test_grounding_verifier_llm_eval():
    mock_model = SimpleNamespace(
        invoke=lambda prompt: SimpleNamespace(
            content='{"is_grounded": true, "grounding_score": 0.88, "unsupported_claims": [], "cited_chunk_ids": ["c1"]}'
        )
    )
    verifier = GroundingVerifier(weak_model=mock_model, enable_eval=True)
    citations = [{"chunk_id": "c1", "rerank_score": 0.9}]
    res = verifier.verify(
        answer="Nikon Zfc có cổng Type-C [c1].",
        context_text="Nikon Zfc features USB Type-C connector.",
        citations=citations,
    )
    assert res["is_grounded"] is True
    assert res["grounding_score"] == 0.88
