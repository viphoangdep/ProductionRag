"""RAG Pipeline: orchestrates two-stage hybrid retrieval, HyDE expansion, RRF fusion, and citation packing."""
from __future__ import annotations

import json
import logging
from typing import Any

from config.settings import settings
from core.rag.context_builder import build_context, depth_of
from core.rag.types import RAGRuntimeContext, SearchFilter
from util.fusion import (
    aggregate_parents,
    is_evidence_sufficient,
    rrf_fuse,
    top_parents_expand,
)
from util.string_util import normalize_ws

log = logging.getLogger("ai-service")


def pack_citation_artifact(
    top: list[tuple[Any, float]],
    query_norm: str,
    paraphrases: list[str],
    branch: str,
    stage1: tuple[float, float],
    assets: Any,
    max_table_rows: int,
    sections: Any = None,
) -> tuple[str, dict[str, Any]]:
    """Build formatted context string and citation artifact dictionary."""
    top_docs = [doc for doc, _ in top]
    context = build_context(top_docs)
    citations = []
    for doc, score in top:
        m = doc.metadata
        extra = assets.resolve_for_chunk(m, max_table_rows)
        # Citation = stored metadata without raw images/tables + depth + rerank score + resolved assets
        citation = {k: v for k, v in m.items() if k not in ("images", "tables")}
        citation["depth"] = depth_of(doc, sections)
        citation["rerank_score"] = float(score)
        citation["images"] = extra.get("images", [])
        citation["tables"] = extra.get("tables", [])
        citations.append(citation)
    artifact = {
        "query_norm": query_norm,
        "paraphrases": paraphrases,
        "branch": branch,
        "citations": citations,
        "stage1_top1": stage1[0],
        "stage1_mean": stage1[1],
    }
    log.info(
        "search done branch=%s citations=%d ctx_chars=%d q=%.60s",
        branch,
        len(citations),
        len(context),
        query_norm,
    )
    return context, artifact


def _resolve_filter(section: str | None, metadata: dict[str, Any] | None, sections: Any) -> SearchFilter | None:
    """Combine section subtree filter and metadata match conditions into a SearchFilter."""
    node_id = None
    if section and sections is not None:
        node_id = sections.resolve(section)
    meta = metadata or {}
    if node_id or meta:
        return SearchFilter(node_id=node_id, metadata=meta)
    return None


class RAGPipeline:
    """Orchestrates document retrieval with confidence-gated HyDE fallback."""

    @staticmethod
    def execute(
        query: str,
        section: str | None,
        metadata: dict[str, Any] | None,
        ctx: RAGRuntimeContext | dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        # Check and consume tool budget
        tool_budget = ctx.tool_budget if isinstance(ctx, RAGRuntimeContext) else ctx.get("tool_budget")
        if tool_budget is not None:
            tool_budget.consume("search_documents")

        retriever = ctx.retriever if isinstance(ctx, RAGRuntimeContext) else ctx["retriever"]
        reranker = ctx.reranker if isinstance(ctx, RAGRuntimeContext) else ctx["reranker"]
        assets = ctx.assets if isinstance(ctx, RAGRuntimeContext) else ctx["assets"]
        sections = ctx.sections if isinstance(ctx, RAGRuntimeContext) else ctx.get("sections")
        cfg = ctx.cfg if isinstance(ctx, RAGRuntimeContext) else ctx.get("cfg", {})

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
        cache = ctx.search_cache if isinstance(ctx, RAGRuntimeContext) else ctx.get("search_cache")
        cache_key = (query_norm, section, json.dumps(metadata, sort_keys=True, default=str))
        if cache is not None and cache_key in cache:
            log.info("search cache hit q=%.60s", query_norm)
            return cache[cache_key]

        filter_spec = _resolve_filter(section, metadata, sections)
        log.info(
            "search start q=%.60s section=%s direct_k=%d thr=top1>=%.2f|mean>=%.2f",
            query_norm,
            section,
            direct_k,
            top1_thr,
            mean_thr,
        )

        # Stage 1: direct retrieve -> aggregate parents -> expand -> rerank
        base = retriever.search(query_norm, direct_k, filter=filter_spec)
        groups, parent_scores = aggregate_parents(base)
        expanded = top_parents_expand(groups, parent_scores, top_parents, max_child)
        reranked_direct = reranker.rerank(query_norm, expanded) if expanded else []
        top_direct = reranked_direct[:top_n]
        s = [float(x[1]) for x in top_direct]
        stage1 = (s[0] if s else 0.0, sum(s) / len(s) if s else 0.0)
        log.info(
            "search stage1 base=%d groups=%d expanded=%d top1=%.3f mean=%.3f q=%.60s",
            len(base),
            len(groups),
            len(expanded),
            *stage1,
            query_norm,
        )

        # Early exit gate
        if is_evidence_sufficient(reranked_direct, top_n, top1_thr, mean_thr):
            log.info("search gate=TRUE -> branch=direct q=%.60s", query_norm)
            res = pack_citation_artifact(
                top_direct, query_norm, [], "direct", stage1, assets, max_table_rows, sections
            )
            if cache is not None:
                cache[cache_key] = res
            return res

        # Stage 2 fallback: HyDE paraphrases -> multi-search -> RRF fusion -> expand -> rerank
        log.info(
            "search gate=FALSE -> branch=hyde_fallback top1=%.3f mean=%.3f q=%.60s",
            *stage1,
            query_norm,
        )
        paraphrase_fn = ctx.paraphrase if isinstance(ctx, RAGRuntimeContext) else ctx["paraphrase"]
        paraphrases = paraphrase_fn(query_norm) if paraphrase_fn else []
        log.info("search hyde n=%d q=%.60s", len(paraphrases), query_norm)

        rank_lists = {"__base__": base}
        for q in paraphrases:
            rank_lists[q] = retriever.search(q, hyde_k, filter=filter_spec)

        fused = rrf_fuse(rank_lists, rrf_k=rrf_k, fused_k=fused_k)
        groups, parent_scores = aggregate_parents(fused)
        expanded = top_parents_expand(groups, parent_scores, top_parents, max_child)
        reranked = reranker.rerank(query_norm, expanded)
        top = reranked[:top_n]
        fs = [float(x[1]) for x in top]
        log.info(
            "search stage2 lists=%d fused=%d expanded=%d top1=%.3f mean=%.3f q=%.60s",
            len(rank_lists),
            len(fused),
            len(expanded),
            fs[0] if fs else 0.0,
            sum(fs) / len(fs) if fs else 0.0,
            query_norm,
        )

        res = pack_citation_artifact(
            top, query_norm, paraphrases, "hyde_fallback", stage1, assets, max_table_rows, sections
        )
        if cache is not None:
            cache[cache_key] = res
        return res
