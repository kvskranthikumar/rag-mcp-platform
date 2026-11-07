"""Embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Common interface for local and hosted embedding models."""

    name: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for storage/indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query for retrieval."""
