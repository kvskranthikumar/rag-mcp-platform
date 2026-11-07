"""Local LLM provider: Ollama."""

from __future__ import annotations

from .base import LLMProvider


class OllamaLLMProvider(LLMProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        import httpx

        self.name = f"ollama:{model}"
        self._model = model
        self._client = httpx.Client(base_url=base_url, timeout=120.0, trust_env=False)

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = self._client.post(
            "/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "options": {"temperature": temperature},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
