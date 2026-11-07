"""Shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: str
    content: str
    heading: str | None
    url_anchor: str | None
    title: str | None
    source_path: str
    url: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def citation(self) -> str:
        """Human-readable source label for grounding answers."""
        label = self.title or self.source_path
        if self.heading:
            label = f"{label} > {self.heading}"
        page_number = self.metadata.get("page_number")
        if type(page_number) is int and page_number > 0:
            label = f"{label} (page {page_number})"
        if self.url:
            anchor = f"#{self.url_anchor}" if self.url_anchor else ""
            label = f"{label} ({self.url}{anchor})"
        return label
