"""``GET /health`` route for the ingestion UI (Container Apps / K8s probes).

Registered via ``st.App(routes=...)`` in :mod:`rag_mcp_platform.ingest_ui.launcher`.
Liveness only — it does not check PostgreSQL or Azure OpenAI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.routing import BaseRoute


async def _health(_request):  # noqa: ANN001, ANN202
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})


def health_routes(base_url_path: str = "") -> list[BaseRoute]:
    """Routes to pass to ``st.App(routes=...)``.

    ``/health`` is always served so in-container probes work. When the app runs
    behind a reverse proxy that keeps a path prefix (``server.baseUrlPath``),
    ``/<prefix>/health`` is served as well so external probes hit the same route.
    """
    from starlette.routing import Route

    paths = ["/health"]
    prefix = base_url_path.strip("/")
    if prefix:
        paths.append(f"/{prefix}/health")

    return [Route(path, _health, methods=["GET", "HEAD"]) for path in paths]
