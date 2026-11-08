"""Abstract interface for section tree and hierarchical document structure resolvers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from core.rag.types import SearchFilter


class BaseSectionStore(ABC):
    """Interface for hierarchical section/chapter resolution and tree depth calculation."""

    @abstractmethod
    def resolve(self, name: str) -> str | None:
        """Resolve a human-readable section/chapter name to a unique node ID."""
        raise NotImplementedError

    @abstractmethod
    def depth_of(self, doc: Document) -> int:
        """Calculate the tree depth of a given document chunk."""
        raise NotImplementedError

    @abstractmethod
    def subtree_filter(self, node_id: str) -> SearchFilter:
        """Generate a provider-agnostic filter that includes the node and all its descendants."""
        raise NotImplementedError

