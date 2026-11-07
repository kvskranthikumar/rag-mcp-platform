"""LLM provider factory (config-driven)."""

from __future__ import annotations

from functools import lru_cache
from ipaddress import ip_address
from urllib.parse import urlsplit

from ...settings import get_settings
from ...settings import require_external_data_permission
from .base import LLMProvider


def _is_loopback_endpoint(endpoint: str) -> bool:
    try:
        host = urlsplit(endpoint).hostname
        return host == "localhost" or bool(host and ip_address(host).is_loopback)
    except ValueError:
        return False


@lru_cache
def get_llm_provider() -> LLMProvider:
    s = get_settings()
    provider = s.llm_provider.lower()

    if provider == "openai":
        require_external_data_permission(s, "OpenAI chat")
        from .openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(
            model=s.llm_model,
            api_key=s.openai_api_key,
            base_url=s.openai_base_url,
        )

    if provider == "azure":
        require_external_data_permission(s, "Azure OpenAI chat")
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
        require_external_data_permission(s, "Google Gemini")
        from .gemini_provider import GeminiLLMProvider

        return GeminiLLMProvider(model=s.llm_model, api_key=s.google_api_key)

    if provider == "ollama":
        if not _is_loopback_endpoint(s.ollama_base_url):
            require_external_data_permission(s, "Remote Ollama")
        from .ollama_provider import OllamaLLMProvider

        return OllamaLLMProvider(model=s.llm_model, base_url=s.ollama_base_url)

    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider!r}")


__all__ = ["LLMProvider", "get_llm_provider"]
