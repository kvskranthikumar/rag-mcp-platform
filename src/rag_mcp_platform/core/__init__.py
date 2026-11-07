"""Shared retrieval core (used by both the MCP server and the web app)."""

from .search import answer, search

__all__ = ["search", "answer"]
