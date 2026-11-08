"""Abstract interface for asset resolvers (images, table rows, external files)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAssetStore(ABC):
    """Interface for resolving chunk assets (CSV tables, images) and retrieving raw asset bytes."""

    @abstractmethod
    def resolve_for_chunk(
        self,
        metadata: dict[str, Any],
        max_rows: int = 50,
    ) -> dict[str, Any]:
        """Resolve images and CSV table data associated with a chunk's metadata.

        Returns:
            A dictionary with 'images': list[dict] and 'tables': list[dict].
        """
        raise NotImplementedError

    @abstractmethod
    def image_bytes(self, image_id: str) -> tuple[bytes, str]:
        """Retrieve the binary payload and MIME media type of a referenced image.

        Returns:
            (data_bytes, mime_type)
        """
        raise NotImplementedError
