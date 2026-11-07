"""MCP server: documentation search for Cursor / Claude Desktop.

Supports stdio (local subprocess) and streamable HTTP (hosted). The host LLM
composes the final answer from retrieved chunks (retrieval-only RAG).
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from psycopg.rows import dict_row
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..core.search import search
from ..db import get_pool
from ..settings import get_settings
from .auth import ApiKeyMiddleware


def build_mcp() -> FastMCP:
    """Create a FastMCP instance with tools, resources, and HTTP settings."""
    settings = get_settings()

    transport_security = None
    if settings.mcp_host in ("127.0.0.1", "localhost", "::1"):
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
        )

    mcp = FastMCP(
        "RAG MCP Platform",
        instructions=(
            "Search the documentation indexed by this server. For questions grounded "
            "in indexed documents, call search_documentation before answering and "
            "cite the returned source labels. Do not invent documentation details "
            "when retrieval returns no relevant passages."
        ),
        host=settings.mcp_host,
        port=settings.mcp_port,
        stateless_http=settings.mcp_stateless_http,
        retry_interval=settings.mcp_sse_retry_interval_ms,
        transport_security=transport_security,
    )

    @mcp.tool()
    def search_documentation(query: str, top_k: int = 8) -> str:
        """Search indexed documentation and return relevant passages with citations.

        Call this tool when a question depends on documents indexed by this server.
        Base documentation claims on the returned passages and cite their source
        labels. If no relevant passages are found, explain that the index does not
        contain enough information to answer.

        Args:
            query: Natural-language question or keywords about the indexed documents.
            top_k: Maximum number of passages to return (default 8).
        """
        results = search(query, top_k)
        if not results:
            return (
                "No relevant indexed documentation was found for this query."
            )

        blocks: list[str] = []
        for i, r in enumerate(results, 1):
            url_line = f"\nURL: {r.url}" if r.url else ""
            blocks.append(
                f"### Result {i} (relevance {r.score:.3f})\n"
                f"Source: {r.citation()}{url_line}\n\n{r.content}"
            )
        return "\n\n---\n\n".join(blocks)

    @mcp.resource("docs://documents")
    def list_documents() -> str:
        """List indexed documents (title, type, and source path)."""
        pool = get_pool()
        with pool.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute(
                "SELECT d.title, d.doc_type, d.source_path, count(c.id) AS chunk_count "
                "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
                "GROUP BY d.id ORDER BY d.title"
            )
            rows = cur.fetchall()

        if not rows:
            return "No documents indexed yet. Run `rag-mcp-platform ingest`."

        lines = [f"{len(rows)} documents indexed:\n"]
        for r in rows:
            lines.append(
                f"- {r['title']}  [{r['doc_type']}, {r['chunk_count']} chunks]  "
                f"({r['source_path']})"
            )
        return "\n".join(lines)

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return mcp


def main_stdio() -> None:
    get_settings()
    build_mcp().run(transport="stdio")


def _require_http_auth(settings) -> None:
    host = settings.mcp_host.strip("[]").lower()
    if host not in {"127.0.0.1", "localhost", "::1"} and not settings.mcp_api_key:
        raise PermissionError(
            "MCP_API_KEY is required when the MCP HTTP server binds to a non-loopback host."
        )


def main_http() -> None:
    import uvicorn

    settings = get_settings()
    _require_http_auth(settings)
    mcp = build_mcp()
    app = mcp.streamable_http_app()
    if settings.mcp_api_key:
        app = ApiKeyMiddleware(app, settings.mcp_api_key)

    print(
        f"RAG MCP Platform MCP (streamable HTTP) listening on "
        f"http://{settings.mcp_host}:{settings.mcp_port}/mcp"
    )
    uvicorn.run(app, host=settings.mcp_host, port=settings.mcp_port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG MCP Platform server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="stdio for Cursor subprocess (default); http for hosted streamable HTTP.",
    )
    args = parser.parse_args()
    if args.transport == "http":
        main_http()
    else:
        main_stdio()


if __name__ == "__main__":
    main()
