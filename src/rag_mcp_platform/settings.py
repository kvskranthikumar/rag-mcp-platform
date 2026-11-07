"""Central configuration, loaded from environment / .env file."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from .config import resolve_env_file

_env_file = resolve_env_file()
_settings_kwargs: dict = {
    "env_file_encoding": "utf-8",
    "extra": "ignore",
    "case_sensitive": False,
}
if _env_file is not None:
    _settings_kwargs["env_file"] = _env_file


class Settings(BaseSettings):
    model_config = SettingsConfigDict(**_settings_kwargs)

    # --- Database -----------------------------------------------------------
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_mcp_platform"

    # --- Embeddings ---------------------------------------------------------
    embedding_provider: str = "azure"
    # Base model name (used to detect `dimensions` support). For Azure the actual
    # request targets AZURE_EMBEDDING_DEPLOYMENT below.
    embedding_model: str = "text-embedding-3-large"
    # 1536 keeps us within pgvector's 2000-dim HNSW limit. text-embedding-3-large
    # supports reduced dimensions via the `dimensions` API parameter (Matryoshka).
    embedding_dim: int = 1536
    embedding_batch_size: int = 32
    embedding_max_input_tokens: int = 8191
    embedding_batch_max_tokens: int = 8192
    embedding_max_retries: int = 2
    embedding_timeout_seconds: float = 60.0
    embedding_reuse_enabled: bool = False
    chunk_insert_batch_size: int = 200
    langfuse_enabled: bool = False
    # Optional model/version tag. Azure encapsulates the model version in the
    # deployment, so the API does not need it; this value is recorded with each
    # document to detect embedding-space changes (see embedding_signature).
    embedding_version: str = ""

    # --- Answer LLM ---------------------------------------------------------
    llm_provider: str = "azure"
    llm_model: str = "gpt-4o-mini"  # label only; Azure routes via the deployment
    llm_temperature: float = 0.1

    # --- Azure OpenAI service credentials ----------------------------------
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-02-01"
    azure_embedding_deployment: str | None = None
    azure_llm_deployment: str | None = None

    # --- Other providers (kept for future swap; unused with Azure) ----------
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    google_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # --- Retrieval / hybrid search -----------------------------------------
    top_k: int = 8
    candidate_count: int = 50
    rrf_k: int = 60
    rerank_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Chunking / ingestion ----------------------------------------------
    chunk_max_tokens: int = 700
    chunk_overlap_tokens: int = 100
    ingestion_pipeline_version: str = "10"
    docs_source_dir: str = "docs-source"
    pdf_extract_tables: bool = True
    pdf_remove_repeated_margins: bool = False

    # --- Hosted MCP (streamable HTTP) --------------------------------------
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8080
    # Optional bearer token for ``Authorization: Bearer …`` on the HTTP endpoint.
    mcp_api_key: str | None = None
    # Stateful sessions (default) let VS Code / Copilot reconnect SSE after idle
    # drops. Use stateless only behind a gateway with sticky sessions disabled and
    # accept weaker client reconnect behavior.
    mcp_stateless_http: bool = False
    # Optional SSE ``retry`` field (milliseconds) sent to clients on stream open.
    mcp_sse_retry_interval_ms: int | None = None

    @property
    def processing_fingerprint(self) -> str:
        """Identify the parsing, cleaning, and chunking configuration."""
        configuration = {
            "pipeline_version": self.ingestion_pipeline_version,
            "chunk_max_tokens": self.chunk_max_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "pdf_extract_tables": self.pdf_extract_tables,
            "pdf_remove_repeated_margins": self.pdf_remove_repeated_margins,
        }
        serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def embedding_signature(self) -> str:
        """Identify the embedding space; changing it requires an isolated rebuild."""
        parts = [self.embedding_provider, self.embedding_model, str(self.embedding_dim)]
        if self.embedding_version:
            parts.append(self.embedding_version)
        return ":".join(parts)


@lru_cache
def get_settings() -> Settings:
    return Settings()
