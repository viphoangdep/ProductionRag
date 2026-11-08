"""Abstract interface for grounding and hallucination verification."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGroundingVerifier(ABC):
    """Interface for verifying answer factual correctness against retrieved context."""

    @abstractmethod
    def verify(
        self,
        answer: str,
        context_text: str,
        citations: list[dict[str, Any]],
        is_simple_query: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Verify the factual alignment of the answer with citations and context.

        Returns:
            Dictionary containing verification results (is_grounded, grounding_score, etc.).
        """
        raise NotImplementedError
