"""FastAPI backend for the documentation chat UI.

Reuses the shared retrieval core: ``/api/search`` returns hybrid-search chunks,
``/api/ask`` runs the full RAG flow (retrieve + grounded LLM answer with
citations). The static chat UI is served from ``web/static``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from psycopg.rows import dict_row
from pydantic import BaseModel

from ..core.search import answer, search
from ..db import get_pool
from ..settings import get_settings

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="RAG MCP Platform", description="Grounded Q&A over indexed documents")


class AskRequest(BaseModel):
    query: str
    top_k: int | None = None


class Source(BaseModel):
    citation: str
    source_path: str
    url: str | None = None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class SearchResult(BaseModel):
    citation: str
    source_path: str
    url: str | None = None
    heading: str | None = None
    content: str
    score: float


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "embedding_provider": s.embedding_provider,
        "embedding_model": s.embedding_model,
        "embedding_dim": s.embedding_dim,
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "rerank_enabled": s.rerank_enabled,
    }


@app.post("/api/ask", response_model=AskResponse)
def api_ask(req: AskRequest) -> AskResponse:
    result = answer(req.query, req.top_k)
    return AskResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
    )


@app.post("/api/search", response_model=list[SearchResult])
def api_search(req: AskRequest) -> list[SearchResult]:
    return [
        SearchResult(
            citation=c.citation(),
            source_path=c.source_path,
            url=c.url,
            heading=c.heading,
            content=c.content,
            score=c.score,
        )
        for c in search(req.query, req.top_k)
    ]


@app.get("/api/documents")
def api_documents() -> dict:
    pool = get_pool()
    with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            "SELECT d.title, d.doc_type, d.source_path, d.url, count(c.id) AS chunk_count "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
            "GROUP BY d.id ORDER BY d.title"
        )
        rows = cur.fetchall()
    return {"count": len(rows), "documents": rows}


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    import uvicorn

    get_settings()  # fail fast on misconfiguration
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
