"""Abstract interface for query expansion (HyDE, multi-query, paraphrase, etc.)."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseQueryExpander(ABC):
    """Interface for generating query paraphrases or hypothetical document expansions."""

    @abstractmethod
    def expand(self, query: str, n: int = 3) -> list[str]:
        """Expand a query into n alternative formulations.

        Args:
            query: The original query text.
            n: Number of expanded variants requested.

        Returns:
            List of generated query strings.
        """
        raise NotImplementedError

