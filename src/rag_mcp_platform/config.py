"""Resolve configuration sources for local vs container runtimes."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root() -> Path | None:
    """Return the repository root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def resolve_env_file() -> Path | None:
    """Pick the dotenv file for Settings, or None for OS environment only.

    - ``RAG_MCP_PLATFORM_RUNTIME=container`` — no dotenv; use injected OS env vars.
    - ``RAG_MCP_PLATFORM_ENV_FILE`` — explicit path (empty string disables dotenv).
    - Otherwise — project-root ``.env`` when present, else cwd ``.env``.
    """
    runtime = os.environ.get("RAG_MCP_PLATFORM_RUNTIME", "local").strip().lower()
    if runtime == "container":
        return None

    explicit = os.environ.get("RAG_MCP_PLATFORM_ENV_FILE")
    if explicit is not None:
        path = Path(explicit)
        return path if str(explicit).strip() and path.is_file() else None

    root = find_project_root()
    if root is not None:
        project_env = root / ".env"
        if project_env.is_file():
            return project_env

    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        return cwd_env

    return None
