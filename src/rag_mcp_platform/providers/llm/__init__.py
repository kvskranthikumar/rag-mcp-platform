"""LLM provider factory (config-driven)."""

from __future__ import annotations

from functools import lru_cache

from ...settings import get_settings
from .base import LLMProvider


@lru_cache
def get_llm_provider() -> LLMProvider:
    s = get_settings()
    provider = s.llm_provider.lower()

    if provider == "openai":
        from .openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(
            model=s.llm_model,
            api_key=s.openai_api_key,
            base_url=s.openai_base_url,
        )

    if provider == "azure":
        from .openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(
            model=s.llm_model,
            api_key=s.azure_openai_api_key,
            azure=True,
            azure_endpoint=s.azure_openai_endpoint,
            azure_api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_llm_deployment,
        )

    if provider == "gemini":
        from .gemini_provider import GeminiLLMProvider

        return GeminiLLMProvider(model=s.llm_model, api_key=s.google_api_key)

    if provider == "ollama":
        from .ollama_provider import OllamaLLMProvider

        return OllamaLLMProvider(model=s.llm_model, base_url=s.ollama_base_url)

    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider!r}")


__all__ = ["LLMProvider", "get_llm_provider"]
