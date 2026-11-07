"""LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface for hosted and local answer-generation models."""

    name: str

    @abstractmethod
    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        """Return a single completion for the given system + user prompt."""
