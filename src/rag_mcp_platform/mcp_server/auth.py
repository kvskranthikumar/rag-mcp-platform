"""Optional bearer-token auth for the hosted MCP HTTP endpoint."""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class ApiKeyMiddleware:
    """Require ``Authorization: Bearer <key>`` except on exempt paths (e.g. /health)."""

    def __init__(
        self,
        app: ASGIApp,
        api_key: str,
        *,
        exempt_paths: frozenset[str] = frozenset({"/health"}),
    ) -> None:
        self.app = app
        self.api_key = api_key
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode("latin-1")
        if not auth.startswith("Bearer ") or auth[7:] != self.api_key:
            response = JSONResponse({"detail": "Unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
