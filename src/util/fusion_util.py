"""Pure algorithmic operations and helper formulas for RAG fusion:
RRF (Reciprocal Rank Fusion), parent aggregation, child expansion, and evidence gating.

Contains zero external I/O or database dependencies; easily unit-testable.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def rrf_fuse(
    rank_lists: dict[str, list[tuple[Any, float]]],
    rrf_k: int = 60,
    fused_k: int = 20,
) -> list[tuple[Any, float]]:
    """Reciprocal Rank Fusion (RRF) merging ranked lists from multiple queries."""
    scores: dict[str, float] = defaultdict(float)
    doc_by_id: dict[str, Any] = {}
    for _branch, results in rank_lists.items():
        for rank, (doc, _s) in enumerate(results, start=1):
            cid = doc.metadata["chunk_id"]
            scores[cid] += 1.0 / (rrf_k + rank)
            if cid not in doc_by_id:
                doc_by_id[cid] = doc
    ranked = sorted(((doc_by_id[c], s) for c, s in scores.items()), key=lambda x: x[1], reverse=True)
    return ranked[:fused_k]


def group_key(doc: Any) -> str:
    """Extract section identifier for chunk siblings, with fallback to parent_id."""
    return doc.metadata.get("section_id") or doc.metadata["parent_id"]


def group_title(doc: Any) -> str:
    """Extract section title with fallback to parent_title."""
    return doc.metadata.get("section_title") or doc.metadata.get("parent_title") or ""


def aggregate_parents(
    fused: list[tuple[Any, float]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Group fused chunks by parent section and calculate aggregate scores."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc, score in fused:
        groups[group_key(doc)].append({"doc": doc, "score": score})
    parent_scores: dict[str, dict[str, Any]] = {}
    for pid, items in groups.items():
        ss = [i["score"] for i in items]
        parent_scores[pid] = {
            "title": group_title(items[0]["doc"]),
            "max_score": max(ss),
            "mean_score": sum(ss) / len(ss),
            "count": len(ss),
        }
    return groups, parent_scores


def top_parents_expand(
    groups: dict[str, list[dict[str, Any]]],
    parent_scores: dict[str, dict[str, Any]],
    top_parents: int = 3,
    max_child: int = 10,
) -> list[Any]:
    """Select top parent sections and expand their child chunks sorted by chunk_index."""
    ranked = sorted(
        parent_scores.items(),
        key=lambda kv: (kv[1]["max_score"], kv[1]["mean_score"], kv[1]["count"]),
        reverse=True,
    )[:top_parents]
    expanded = []
    for pid, _ in ranked:
        children = sorted(groups[pid], key=lambda it: it["doc"].metadata.get("chunk_index", 0))
        if len(children) > max_child:
            children = sorted(children, key=lambda it: it["score"], reverse=True)[:max_child]
            children = sorted(children, key=lambda it: it["doc"].metadata.get("chunk_index", 0))
        expanded.extend(it["doc"] for it in children)
    return expanded


def is_evidence_sufficient(
    reranked: list[tuple[Any, float]],
    top_n: int,
    top1_thr: float,
    mean_thr: float,
) -> bool:
    """Gate following initial rerank: returns True to bypass HyDE expansion if confidence is sufficient."""
    if not reranked:
        return False
    scores = [float(s) for _, s in reranked[:top_n]]
    if not scores:
        return False
    if scores[0] >= top1_thr:
        return True
    return (sum(scores) / len(scores)) >= mean_thr

