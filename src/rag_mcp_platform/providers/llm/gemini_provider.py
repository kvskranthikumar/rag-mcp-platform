"""Hosted LLM provider: Google Gemini."""

from __future__ import annotations

from .base import LLMProvider


class GeminiLLMProvider(LLMProvider):
    def __init__(self, model: str, api_key: str | None) -> None:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.name = f"gemini:{model}"
        self._model_id = model
        self._genai = genai

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        model = self._genai.GenerativeModel(
            self._model_id,
            system_instruction=system,
        )
        resp = model.generate_content(
            user,
            generation_config={"temperature": temperature},
        )
        return resp.text or ""
