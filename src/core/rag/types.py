"""Data contracts and runtime context for RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SearchFilter:
    """Provider-agnostic filter specification for document retrieval."""
    node_id: str | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGRuntimeContext:
    """Strongly-typed runtime context for the RAG search tool and pipeline."""
    retriever: Any
    reranker: Any
    assets: Any
    sections: Any = None
    weak_model: Any = None
    strong_model: Any = None
    weak_name: str = ""
    strong_name: str = ""
    cfg: dict[str, Any] = field(default_factory=dict)
    search_cache: dict[tuple, Any] = field(default_factory=dict)
    paraphrase: Callable[[str], list[str]] | None = None
    tool_budget: Any = None
    role: str | None = None
    session_id: str = ""
    standalone_query: str | None = None
