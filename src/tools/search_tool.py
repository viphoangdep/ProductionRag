"""LangChain tool adapter for document retrieval."""
from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from core.rag.context_builder import build_context, depth_of, section_label
from core.rag.fusion import (
    aggregate_parents,
    group_key,
    group_title,
    is_evidence_sufficient,
    rrf_fuse,
    top_parents_expand,
)
from core.rag.pipeline import RAGPipeline, pack_citation_artifact

# Backwards compatibility alias for test suites
_pack = pack_citation_artifact


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
    """
    return RAGPipeline.execute(query, section, metadata, runtime.context)
