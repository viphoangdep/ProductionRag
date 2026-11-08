"""Abstract interface for Query Routing and Intent Classification."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(str, Enum):
    """Semantic intent classification of a user query."""

    CHITCHAT = "CHITCHAT"        # Greetings, bot identity, polite closing
    RAG_SEARCH = "RAG_SEARCH"    # Factual product information requiring manual retrieval
    DIRECT_QA = "DIRECT_QA"      # General question answerable without document search


@dataclass
class RouteDecision:
    """Decision made by the query router regarding model tier, tools, and prompts."""

    intent: QueryIntent
    model_tier: str = "strong"  # "weak" or "strong"
    tools_allowed: list[str] = field(default_factory=list)
    system_prompt: str = ""
    confidence: float = 1.0
    reason: str = ""


class BaseQueryRouter(ABC):
    """Interface for classifying user queries and selecting appropriate runtime parameters."""

    @abstractmethod
    def classify_intent(self, query: str, **kwargs: Any) -> QueryIntent:
        """Classify user query into a semantic QueryIntent.

        Args:
            query: The user input text.

        Returns:
            The determined QueryIntent.
        """
        raise NotImplementedError

    @abstractmethod
    def route(self, query: str, context: dict[str, Any] | None = None) -> RouteDecision:
        """Produce a complete RouteDecision for model, prompt, and tools.

        Args:
            query: The user input text.
            context: Runtime context (models, configs, session).

        Returns:
            A RouteDecision object.
        """
        raise NotImplementedError

