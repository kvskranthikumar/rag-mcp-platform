"""Local embedding provider (sentence-transformers): bge / e5 / nomic."""

from __future__ import annotations

from collections.abc import Sequence

from .base import EmbeddingProvider


def _instruction_prefixes(model: str) -> tuple[str, str]:
    """Return (query_prefix, passage_prefix) required by the model family."""
    m = model.lower()
    if "bge" in m:
        # BGE v1.5 recommends a query instruction; passages need no prefix.
        return ("Represent this sentence for searching relevant passages: ", "")
    if "e5" in m:
        return ("query: ", "passage: ")
    if "nomic" in m:
        return ("search_query: ", "search_document: ")
    return ("", "")


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str, dimension: int, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model
        self.name = f"local:{model}"
        self.batch_size = batch_size
        self._model = SentenceTransformer(model, trust_remote_code=True)
        self._q_prefix, self._d_prefix = _instruction_prefixes(model)

        actual_dim = self._model.get_sentence_embedding_dimension()
        self.dimension = actual_dim
        if dimension and dimension != actual_dim:
            raise ValueError(
                f"EMBEDDING_DIM={dimension} does not match model '{model}' "
                f"which outputs {actual_dim} dimensions. Update EMBEDDING_DIM, then "
                f"re-run `rag-mcp-platform init-db --drop` and re-ingest."
            )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = [self._d_prefix + t for t in texts]
        vecs = self._model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            [self._q_prefix + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vec.tolist()
