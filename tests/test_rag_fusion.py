"""Unit tests for pure RAG algorithms: RRF fusion, parent aggregation, gating, and depth."""
from types import SimpleNamespace
import pytest

from core.rag.fusion import (
    rrf_fuse,
    group_key,
    group_title,
    aggregate_parents,
    top_parents_expand,
    is_evidence_sufficient,
)
from core.rag.context_builder import section_label, depth_of, build_context
import util.fusion as util_fusion
import util.fusion_util as util_fusion_util


def test_fusion_util_exports_match():
    assert util_fusion.rrf_fuse is rrf_fuse
    assert util_fusion.group_key is group_key
    assert util_fusion.group_title is group_title
    assert util_fusion.aggregate_parents is aggregate_parents
    assert util_fusion.top_parents_expand is top_parents_expand
    assert util_fusion.is_evidence_sufficient is is_evidence_sufficient
    assert util_fusion_util.rrf_fuse is rrf_fuse


def _make_doc(chunk_id, section_id=None, parent_id=None, section_title=None, chunk_index=0, page_content="test"):
    meta = {
        "chunk_id": chunk_id,
        "section_id": section_id,
        "parent_id": parent_id or section_id,
        "section_title": section_title,
        "chunk_index": chunk_index,
        "page_nos": [1],
    }
    return SimpleNamespace(metadata=meta, page_content=page_content)


def test_rrf_fuse():
    d1 = _make_doc("c1")
    d2 = _make_doc("c2")
    d3 = _make_doc("c3")

    rank_lists = {
        "q1": [(d1, 0.9), (d2, 0.8)],
        "q2": [(d2, 0.95), (d3, 0.7)],
    }

    fused = rrf_fuse(rank_lists, rrf_k=60, fused_k=3)
    # d2 appears in both lists (rank 2 in q1, rank 1 in q2), so it should rank first
    assert len(fused) == 3
    assert fused[0][0].metadata["chunk_id"] == "c2"
    assert fused[0][1] > fused[1][1]


def test_group_key_and_title():
    doc1 = _make_doc("c1", section_id="sec-1", section_title="Section One")
    assert group_key(doc1) == "sec-1"
    assert group_title(doc1) == "Section One"

    doc2 = _make_doc("c2", parent_id="par-1", section_title=None)
    doc2.metadata["parent_title"] = "Parent One"
    assert group_key(doc2) == "par-1"
    assert group_title(doc2) == "Parent One"


def test_aggregate_parents():
    d1 = _make_doc("c1", section_id="s1")
    d2 = _make_doc("c2", section_id="s1")
    d3 = _make_doc("c3", section_id="s2")

    fused = [(d1, 0.8), (d2, 0.4), (d3, 0.7)]
    groups, parent_scores = aggregate_parents(fused)

    assert len(groups["s1"]) == 2
    assert len(groups["s2"]) == 1
    assert parent_scores["s1"]["max_score"] == pytest.approx(0.8)
    assert parent_scores["s1"]["mean_score"] == pytest.approx(0.6)
    assert parent_scores["s1"]["count"] == 2
    assert parent_scores["s2"]["max_score"] == pytest.approx(0.7)


def test_top_parents_expand():
    d1 = _make_doc("c1", section_id="s1", chunk_index=2)
    d2 = _make_doc("c2", section_id="s1", chunk_index=1)
    d3 = _make_doc("c3", section_id="s2", chunk_index=0)

    groups = {
        "s1": [{"doc": d1, "score": 0.5}, {"doc": d2, "score": 0.9}],
        "s2": [{"doc": d3, "score": 0.7}],
    }
    parent_scores = {
        "s1": {"max_score": 0.9, "mean_score": 0.7, "count": 2},
        "s2": {"max_score": 0.7, "mean_score": 0.7, "count": 1},
    }

    expanded = top_parents_expand(groups, parent_scores, top_parents=1, max_child=5)
    # top_parents=1 picks s1, sorted by chunk_index (d2 index 1 before d1 index 2)
    assert len(expanded) == 2
    assert expanded[0].metadata["chunk_id"] == "c2"
    assert expanded[1].metadata["chunk_id"] == "c1"


def test_is_evidence_sufficient_gate():
    d1 = _make_doc("c1")
    d2 = _make_doc("c2")

    # High top 1 score triggers gate
    reranked_high = [(d1, 0.85), (d2, 0.4)]
    assert is_evidence_sufficient(reranked_high, top_n=2, top1_thr=0.7, mean_thr=0.5) is True

    # Low top 1 but high mean triggers gate
    reranked_mean = [(d1, 0.6), (d2, 0.55)]
    assert is_evidence_sufficient(reranked_mean, top_n=2, top1_thr=0.7, mean_thr=0.5) is True

    # Low scores fail gate
    reranked_low = [(d1, 0.3), (d2, 0.2)]
    assert is_evidence_sufficient(reranked_low, top_n=2, top1_thr=0.7, mean_thr=0.5) is False

    # Empty list fails gate
    assert is_evidence_sufficient([], top_n=2, top1_thr=0.7, mean_thr=0.5) is False


def test_build_context_and_section_label():
    d1 = _make_doc("c1", page_content="Hello world")
    d1.metadata["section_path"] = "Chapter 1 > Section A"
    ctx = build_context([d1])
    assert "[CTX 1 | c1 | pages=[1] | section=Chapter 1 > Section A]" in ctx
    assert "Hello world" in ctx
