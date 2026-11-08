"""Abstract interface for document rerankers (CrossEncoder, Cohere, BGE, ColBERT, etc.)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseReranker(ABC):
    """Provider-agnostic interface for document reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        docs: list[Document],
    ) -> list[tuple[Document, float]]:
        """Rerank candidate documents against query, returning (doc, score) pairs sorted descending.

        Args:
            query: The normalized query string.
            docs: List of candidate documents to score.

        Returns:
            List of (Document, score) tuples ordered from highest to lowest relevance score.
        """
        raise NotImplementedError

