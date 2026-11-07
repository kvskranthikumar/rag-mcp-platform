"""Hosted LLM provider: OpenAI / Azure OpenAI chat completions."""

from __future__ import annotations

from .base import LLMProvider
from ...telemetry import record_usage


class OpenAILLMProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        azure: bool = False,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
        azure_deployment: str | None = None,
    ) -> None:
        if azure:
            from openai import AzureOpenAI

            self.name = f"azure:{azure_deployment or model}"
            self._model = azure_deployment or model
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint or "",
                api_version=azure_api_version or "2024-02-01",
            )
        else:
            from openai import OpenAI

            self.name = f"openai:{model}"
            self._model = model
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = getattr(resp, "usage", None)
        record_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )
        return resp.choices[0].message.content or ""
