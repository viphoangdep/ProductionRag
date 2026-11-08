"""Abstract interface for document retrievers (Qdrant, Weaviate, Milvus, Chroma, etc.)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from core.rag.types import SearchFilter


class BaseRetriever(ABC):
    """Provider-agnostic interface for document retrieval."""

    @abstractmethod
    def search(
        self,
        query: str,
        k: int,
        filter: SearchFilter | None = None,
    ) -> list[Document]:
        """Retrieve top-k documents matching the query, optionally constrained by filter.

        Args:
            query: The search query string.
            k: Maximum number of candidate documents to retrieve.
            filter: Optional provider-agnostic filter specification (node_id, metadata, etc.).

        Returns:
            A list of Document objects.
        """
        raise NotImplementedError

