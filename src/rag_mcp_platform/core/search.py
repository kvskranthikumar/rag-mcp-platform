"""Hybrid retrieval (dense + lexical via RRF) and grounded answer composition."""

from __future__ import annotations

import numpy as np
from psycopg.rows import dict_row

from ..db import get_pool
from ..models import RetrievedChunk
from ..providers.embeddings import get_embedding_provider
from ..settings import get_settings
from ..telemetry import observation, stage

SYSTEM_PROMPT = (
    "You are a helpful assistant for a document question-answering tool. "
    "Answer the user's question using ONLY the provided documentation context. "
    "Cite the source label in brackets after each claim, e.g. [Guide > Setup]. "
    "If the context does not contain the answer, reply exactly: "
    '"I could not find this in the documentation." Do not invent information.'
)


def search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Run hybrid search and (optionally) rerank; return top chunks with citations."""
    settings = get_settings()
    top_k = top_k or settings.top_k
    with observation(
        "retrieval.search",
        metadata={"query_characters": len(query), "top_k": top_k},
    ) as span:
        chunks = _search(query, top_k, settings)
        if span is not None:
            span.update(output={"result_count": len(chunks)})
        return chunks


def _search(query: str, top_k: int, settings) -> list[RetrievedChunk]:
    embedder = get_embedding_provider()
    with stage("query_embedding"):
        query_vec = np.asarray(embedder.embed_query(query), dtype=np.float32)

    # Fetch more candidates when reranking so the cross-encoder has room to work.
    fetch_k = max(top_k, settings.candidate_count) if settings.rerank_enabled else top_k

    pool = get_pool()
    with stage("hybrid_search"), pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            "SELECT * FROM search_hybrid(%s, %s, %s, %s, %s)",
            (query, query_vec, fetch_k, settings.rrf_k, settings.candidate_count),
        )
        rows = cur.fetchall()

    chunks = [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            document_id=str(r["document_id"]),
            content=r["content"],
            heading=r["heading"],
            url_anchor=r["url_anchor"],
            title=r["title"],
            source_path=r["source_path"],
            url=r["url"],
            metadata=r["metadata"] or {},
            score=float(r["score"]),
        )
        for r in rows
    ]

    if settings.rerank_enabled and chunks:
        from .reranker import rerank

        with stage("reranking"):
            chunks = rerank(query, chunks, top_k)

    return chunks[:top_k]


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(
        f"[{c.citation()}]\n{c.content}" for c in chunks
    )


def answer(query: str, top_k: int | None = None) -> dict:
    """Full RAG: retrieve, then generate a grounded answer with the configured LLM."""
    from ..providers.llm import get_llm_provider

    settings = get_settings()
    with observation(
        "rag.answer",
        metadata={"query_characters": len(query), "top_k": top_k or settings.top_k},
    ) as span:
        chunks = search(query, top_k)
        if not chunks:
            result = {"answer": "I could not find this in the documentation.", "sources": []}
        else:
            llm = get_llm_provider()
            context = build_context(chunks)
            user_prompt = f"Question: {query}\n\nDocumentation context:\n{context}"
            with observation(
                "answer.generation",
                as_type="generation",
                model=settings.llm_model,
                model_parameters={"temperature": settings.llm_temperature},
                metadata={
                    "provider": settings.llm_provider,
                    "context_chunk_count": len(chunks),
                },
            ):
                text = llm.complete(
                    SYSTEM_PROMPT, user_prompt, temperature=settings.llm_temperature,
                )
            result = {
                "answer": text,
                "sources": [
                    {
                        "citation": c.citation(),
                        "source_path": c.source_path,
                        "url": c.url,
                        "score": c.score,
                    }
                    for c in chunks
                ],
            }
        if span is not None:
            span.update(output={
                "source_count": len(result["sources"]),
                "answer_characters": len(result["answer"]),
            })
        return result
