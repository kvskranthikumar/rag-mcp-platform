"""Optional cross-encoder reranking of fused candidates."""

from __future__ import annotations

from functools import lru_cache

from ..models import RetrievedChunk
from ..settings import get_settings


@lru_cache
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return chunks
    model = _get_reranker()
    scores = model.predict([(query, c.content) for c in chunks])
    ordered = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    result: list[RetrievedChunk] = []
    for chunk, score in ordered[:top_k]:
        chunk.score = float(score)
        result.append(chunk)
    return result
