"""Factory agent (langgraph) + runtime context cho tool."""
from __future__ import annotations

from config.settings import settings


def create_agent(weak_model, checkpointer=None):
    """Agent dùng model tier 1 + tool search; router middleware chọn model theo câu hỏi."""
    from langchain.agents import create_agent as _create  # lazy: nặng

    from config.prompts import SYSTEM_PROMPT
    from core.router import dynamic_router
    from tools.errors import handle_tool_errors
    from tools.search_tool import search_documents

    return _create(model=weak_model, tools=[search_documents],
                   middleware=[dynamic_router, handle_tool_errors],
                   system_prompt=SYSTEM_PROMPT,
                   checkpointer=checkpointer)


def create_agent_context(*, retriever, reranker, assets, sections,
                         weak_model, strong_model):
    """Context runtime cho tool: adapter + model + tham số pipeline."""
    return {"retriever": retriever, "reranker": reranker,
            "assets": assets, "sections": sections,
            "weak_model": weak_model, "strong_model": strong_model,
            "weak_name": settings.llm_lv1, "strong_name": settings.llm_lv2,
            "cfg": {"hyde_k": settings.hyde_k, "direct_k": settings.direct_k,
                    "evidence_top1_thr": settings.evidence_top1_thr,
                    "evidence_mean_thr": settings.evidence_mean_thr,
                    "fused_k": settings.fused_k,
                    "rrf_k": settings.rrf_k, "top_parents": settings.top_parents,
                    "max_child": settings.max_child_per_parent, "top_n": settings.top_n,
                    "max_table_rows": settings.max_table_rows}}
