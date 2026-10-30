"""LangChain tool: HyDE -> hybrid search -> RRF -> parent-expand -> rerank -> context.

Orchestration tách từ rag_query.ipynb cells 12-23. Các bước thuần (không I/O)
nằm ngay trong file này để đọc pipeline một chỗ.
"""
from typing import Any

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from config.settings import settings
from util.string_util import normalize_ws

import json
import logging
from collections import defaultdict

log = logging.getLogger("ai-service")


# ============================================================
# Bước thuần của pipeline (không I/O)
# ============================================================

def rrf_fuse(rank_lists: dict, rrf_k: int = 60, fused_k: int = 20) -> list:
    scores: dict = defaultdict(float)
    doc_by_id: dict = {}
    for _branch, results in rank_lists.items():
        for rank, (doc, _s) in enumerate(results, start=1):
            cid = doc.metadata["chunk_id"]
            scores[cid] += 1.0 / (rrf_k + rank)
            if cid not in doc_by_id:
                doc_by_id[cid] = doc
    ranked = sorted(((doc_by_id[c], s) for c, s in scores.items()), key=lambda x: x[1], reverse=True)
    return ranked[:fused_k]


def group_key(doc) -> str:
    """Section cua chunk (siblings). Fallback parent_id cho data cu."""
    return doc.metadata.get("section_id") or doc.metadata["parent_id"]


def group_title(doc) -> str:
    return doc.metadata.get("section_title") or doc.metadata.get("parent_title")


def aggregate_parents(fused: list) -> tuple[dict, dict]:
    groups: dict = defaultdict(list)
    for doc, score in fused:
        groups[group_key(doc)].append({"doc": doc, "score": score})
    parent_scores = {}
    for pid, items in groups.items():
        ss = [i["score"] for i in items]
        parent_scores[pid] = {
            "title": group_title(items[0]["doc"]),
            "max_score": max(ss),
            "mean_score": sum(ss) / len(ss),
            "count": len(ss),
        }
    return groups, parent_scores


def top_parents_expand(groups: dict, parent_scores: dict, top_parents: int = 3,
                       max_child: int = 10) -> list:
    ranked = sorted(parent_scores.items(),
                    key=lambda kv: (kv[1]["max_score"], kv[1]["mean_score"], kv[1]["count"]),
                    reverse=True)[:top_parents]
    expanded = []
    for pid, _ in ranked:
        children = sorted(groups[pid], key=lambda it: it["doc"].metadata.get("chunk_index", 0))
        if len(children) > max_child:
            children = sorted(children, key=lambda it: it["score"], reverse=True)[:max_child]
            children = sorted(children, key=lambda it: it["doc"].metadata.get("chunk_index", 0))
        expanded.extend(it["doc"] for it in children)
    return expanded


def build_context(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        parts.append(f"[CTX {i} | {m.get('chunk_id')} | pages={m.get('page_nos')} "
                     f"| section={section_label(m)}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def section_label(m: dict) -> str:
    """Nhãn section đọc được: section_path, fallback về source_file (chunk gốc __ROOT__)."""
    path = m.get("section_path") or ""
    if path and path != "__ROOT__":
        return path
    return m.get("source_file") or m.get("parent_title") or "__ROOT__"


def depth_of(doc, sections=None) -> int:
    """Depth của chunk theo section tree mới (node['level']); fallback từ section_path."""
    m = doc.metadata
    sid = m.get("section_id")
    if sections is not None and sid:
        node = getattr(sections, "nodes", {}).get(sid)
        if node and node.get("level") is not None:
            return int(node["level"])
    path = m.get("section_path") or ""
    if not path or path == "__ROOT__":
        return 0
    return path.count(" > ") + 1


def is_evidence_sufficient(reranked: list, top_n: int, top1_thr: float, mean_thr: float) -> bool:
    """Gate sau rerank nhanh: True = khoi HyDE. reranked = [(doc, score)]."""
    if not reranked:
        return False
    scores = [float(s) for _, s in reranked[:top_n]]
    if not scores:
        return False
    if scores[0] >= top1_thr:
        return True
    return sum(scores) / len(scores) >= mean_thr


# ============================================================
# Tool
# ============================================================

class SearchDocumentsInput(BaseModel):
    """Input for searching documents."""

    query: str = Field(description="The user's question, verbatim.")

    section: str | None = Field(
        default=None,
        description=(
            "Optional chapter/section name (e.g. 'chuong 3') to restrict "
            "search to that subtree. Omit for corpus-wide search."
        ),
    )

    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional exact-match metadata filters "
            '(e.g. {"source_file": "Z5UM.pdf"}). Omit to no filtering.'
        ),
    )


def _metadata_conditions(metadata: dict[str, Any] | None) -> list:
    from qdrant_client import models  # lazy: heavy dep only at runtime

    return [
        models.FieldCondition(key=f"metadata.{k}", match=models.MatchValue(value=v))
        for k, v in (metadata or {}).items()
    ]


def _resolve_filter(section: str | None, metadata: dict[str, Any] | None, sections: Any):
    """Combine section subtree filter (should) + metadata matches (must)."""
    from qdrant_client import models  # lazy: heavy dep only at runtime

    section_filter = None
    if section and sections is not None:
        node_id = sections.resolve(section)
        if node_id:
            section_filter = sections.subtree_filter(node_id)
    must = _metadata_conditions(metadata)
    if section_filter is not None and must:
        return models.Filter(must=must, should=section_filter.should)
    if must:
        return models.Filter(must=must)
    return section_filter


def _pack(top, query_norm, paraphrases, branch, stage1, assets, max_table_rows,
          sections=None) -> tuple[str, dict]:
    top_docs = [doc for doc, _ in top]
    context = build_context(top_docs)
    citations = []
    for doc, score in top:
        m = doc.metadata
        extra = assets.resolve_for_chunk(m, max_table_rows)
        # Citation = metadata đã lưu (chunks_metadata.json) + depth + score + asset đã resolve.
        citation = {k: v for k, v in m.items() if k not in ("images", "tables")}
        citation["depth"] = depth_of(doc, sections)
        citation["rerank_score"] = float(score)
        citation["images"] = extra.get("images", [])
        citation["tables"] = extra.get("tables", [])
        citations.append(citation)
    artifact = {"query_norm": query_norm, "paraphrases": paraphrases,
                "branch": branch, "citations": citations,
                "stage1_top1": stage1[0], "stage1_mean": stage1[1]}
    log.info("search done branch=%s citations=%d ctx_chars=%d q=%.60s",
             branch, len(citations), len(context), query_norm)
    return context, artifact


def _search_impl(query: str, section: str | None, metadata: dict[str, Any] | None,
                 ctx: dict) -> tuple[str, dict]:
    # Check & consume tool budget if provided in context
    tool_budget = ctx.get("tool_budget")
    if tool_budget is not None:
        tool_budget.consume("search_documents")

    retriever = ctx["retriever"]
    reranker = ctx["reranker"]
    assets = ctx["assets"]
    sections = ctx.get("sections")  # optional: corpus cu thieu section_tree van chay
    cfg = ctx["cfg"]

    direct_k = cfg.get("direct_k", settings.direct_k)
    top1_thr = cfg.get("evidence_top1_thr", settings.evidence_top1_thr)
    mean_thr = cfg.get("evidence_mean_thr", settings.evidence_mean_thr)
    hyde_k = cfg.get("hyde_k", settings.hyde_k)
    fused_k = cfg.get("fused_k", settings.fused_k)
    rrf_k = cfg.get("rrf_k", settings.rrf_k)
    top_parents = cfg.get("top_parents", settings.top_parents)
    max_child = cfg.get("max_child", settings.max_child_per_parent)
    top_n = cfg.get("top_n", settings.top_n)
    max_table_rows = cfg.get("max_table_rows", settings.max_table_rows)

    query_norm = normalize_ws(query)
    cache = ctx.get("search_cache")
    cache_key = (query_norm, section, json.dumps(metadata, sort_keys=True, default=str))
    if cache is not None and cache_key in cache:
        log.info("search cache hit q=%.60s", query_norm)
        return cache[cache_key]
    qdrant_filter = _resolve_filter(section, metadata, sections)
    log.info("search start q=%.60s section=%s direct_k=%d thr=top1>=%.2f|mean>=%.2f",
             query_norm, section, direct_k, top1_thr, mean_thr)

    # Stage 1: direct retrieve -> expand -> rerank nhanh
    base = retriever.search(query_norm, direct_k, qdrant_filter)
    groups, parent_scores = aggregate_parents(base)
    expanded = top_parents_expand(groups, parent_scores, top_parents, max_child)
    reranked_direct = reranker.rerank(query_norm, expanded) if expanded else []
    top_direct = reranked_direct[:top_n]
    s = [float(x[1]) for x in top_direct]
    stage1 = (s[0] if s else 0.0, sum(s) / len(s) if s else 0.0)
    log.info("search stage1 base=%d groups=%d expanded=%d top1=%.3f mean=%.3f q=%.60s",
             len(base), len(groups), len(expanded), *stage1, query_norm)
    if is_evidence_sufficient(reranked_direct, top_n, top1_thr, mean_thr):
        log.info("search gate=TRUE -> branch=direct q=%.60s", query_norm)
        res = _pack(top_direct, query_norm, [], "direct", stage1, assets, max_table_rows, sections)
        if cache is not None:
            cache[cache_key] = res
        return res

    # Stage 2 fallback: HyDE x3 -> hybrid x4 -> RRF -> expand -> rerank
    log.info("search gate=FALSE -> branch=hyde_fallback top1=%.3f mean=%.3f q=%.60s",
             *stage1, query_norm)
    paraphrases = ctx["paraphrase"](query_norm)
    log.info("search hyde n=%d q=%.60s", len(paraphrases), query_norm)
    rank_lists = {"__base__": base}
    for q in paraphrases:
        rank_lists[q] = retriever.search(q, hyde_k, qdrant_filter)

    fused = rrf_fuse(rank_lists, rrf_k=rrf_k, fused_k=fused_k)
    groups, parent_scores = aggregate_parents(fused)
    expanded = top_parents_expand(groups, parent_scores, top_parents, max_child)
    reranked = reranker.rerank(query_norm, expanded)
    top = reranked[:top_n]
    fs = [float(x[1]) for x in top]
    log.info("search stage2 lists=%d fused=%d expanded=%d top1=%.3f mean=%.3f q=%.60s",
             len(rank_lists), len(fused), len(expanded),
             fs[0] if fs else 0.0, sum(fs) / len(fs) if fs else 0.0, query_norm)
    res = _pack(top, query_norm, paraphrases, "hyde_fallback", stage1, assets, max_table_rows, sections)
    if cache is not None:
        cache[cache_key] = res
    return res


@tool(args_schema=SearchDocumentsInput, response_format="content_and_artifact")
def search_documents(
    query: str,
    section: str | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    runtime: ToolRuntime,
) -> tuple[str, dict]:
    """Search the product manual for passages answering the user's question.

    Uses HyDE query expansion, hybrid (dense + sparse) retrieval, RRF fusion,
    and CrossEncoder reranking. Returns evidence text plus per-chunk citations.

    Args:
        query: The user's question, verbatim.
        section: Optional chapter/section name (e.g. "chuong 3") to restrict
            search to that subtree. Omit for corpus-wide search.
        metadata: Optional exact-match metadata filters
            (e.g. {"source_file": "Z5UM.pdf"}).
    """
    return _search_impl(query, section, metadata, runtime.context)
