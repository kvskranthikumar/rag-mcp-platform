"""Hosted embedding provider: OpenAI / Azure OpenAI (text-embedding-3-*)."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from .base import EmbeddingProvider
from ...telemetry import increment_metric, observation, record_usage


def _supports_dimensions(model: str) -> bool:
    # Only the v3 models support the `dimensions` truncation parameter.
    return model.startswith("text-embedding-3")


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str,
        dimension: int,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 64,
        *,
        azure: bool = False,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
        azure_deployment: str | None = None,
        max_input_tokens: int = 8191,
        batch_max_tokens: int = 8192,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        import tiktoken

        if min(batch_size, dimension, max_input_tokens, batch_max_tokens) <= 0:
            raise ValueError("Embedding dimensions and batch/token limits must be positive.")
        if max_retries < 0 or not isfinite(timeout) or timeout <= 0:
            raise ValueError("Embedding retries must be nonnegative and timeout finite and positive.")
        self.model_id = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.batch_max_tokens = batch_max_tokens
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

        if azure:
            from openai import AzureOpenAI

            self.name = f"azure:{azure_deployment or model}"
            self._deployment = azure_deployment or model
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint or "",
                api_version=azure_api_version or "2024-02-01",
                max_retries=max_retries,
                timeout=timeout,
            )
        else:
            from openai import OpenAI

            self.name = f"openai:{model}"
            self._deployment = model
            self._client = OpenAI(
                api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout,
            )

    def _create(self, inputs: list[str]) -> list[list[float]]:
        kwargs: dict = {"model": self._deployment, "input": inputs}
        if _supports_dimensions(self.model_id):
            kwargs["dimensions"] = self.dimension
        increment_metric("embedding_sdk_requests")
        with observation(
            "embedding.batch",
            as_type="embedding",
            model=self.model_id,
            metadata={"input_count": len(inputs), "dimensions": self.dimension},
        ) as span:
            resp = self._client.embeddings.create(**kwargs)
            usage = getattr(resp, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            if span is not None:
                record_usage(input_tokens=prompt_tokens, total_tokens=total_tokens)
        usage = getattr(resp, "usage", None)
        for field in ("prompt_tokens", "total_tokens"):
            value = getattr(usage, field, None)
            if type(value) is int and value >= 0:
                increment_metric("embedding_" + field, value)
        data = resp.data
        if (
            len(data) != len(inputs)
            or any(type(item.index) is not int for item in data)
            or sorted(item.index for item in data) != list(range(len(inputs)))
        ):
            raise ValueError("Embedding response indices do not match the requested inputs.")
        return [item.embedding for item in sorted(data, key=lambda item: item.index)]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        unique: dict[str, int] = {}
        for index, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Embedding input {index} must be nonblank text.")
            tokens = len(self._encoding.encode(text, disallowed_special=()))
            if tokens > min(self.max_input_tokens, self.batch_max_tokens):
                raise ValueError(f"Embedding input {index} exceeds the configured token budget.")
            unique.setdefault(text, tokens)
        vectors: dict[str, list[float]] = {}
        batch: list[str] = []
        batch_tokens = 0

        def flush():
            if batch:
                vectors.update(zip(batch, self._create(batch)))

        for text, tokens in unique.items():
            if batch and (len(batch) >= self.batch_size or batch_tokens + tokens > self.batch_max_tokens):
                flush()
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += tokens
        flush()
        return [vectors[text] for text in items]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
