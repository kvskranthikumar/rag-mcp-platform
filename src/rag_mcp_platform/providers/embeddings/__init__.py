"""Embedding provider factory (config-driven)."""

from __future__ import annotations

from functools import lru_cache

from ...settings import get_settings, require_external_data_permission
from .base import EmbeddingProvider


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    s = get_settings()
    provider = s.embedding_provider.lower()

    if provider == "local":
        from .local import LocalEmbeddingProvider

        return LocalEmbeddingProvider(
            model=s.embedding_model,
            dimension=s.embedding_dim,
            batch_size=s.embedding_batch_size,
        )

    hosted_options = {
        "max_input_tokens": s.embedding_max_input_tokens,
        "batch_max_tokens": s.embedding_batch_max_tokens,
        "max_retries": s.embedding_max_retries,
        "timeout": s.embedding_timeout_seconds,
    }

    if provider == "openai":
        require_external_data_permission(s, "OpenAI embeddings")
        from .openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            model=s.embedding_model,
            dimension=s.embedding_dim,
            api_key=s.openai_api_key,
            base_url=s.openai_base_url,
            batch_size=s.embedding_batch_size,
            **hosted_options,
        )

    if provider == "azure":
        require_external_data_permission(s, "Azure OpenAI embeddings")
        from .openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            model=s.embedding_model,
            dimension=s.embedding_dim,
            api_key=s.azure_openai_api_key,
            batch_size=s.embedding_batch_size,
            azure=True,
            azure_endpoint=s.azure_openai_endpoint,
            azure_api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_embedding_deployment,
            **hosted_options,
        )

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {s.embedding_provider!r}")


__all__ = ["EmbeddingProvider", "get_embedding_provider"]
