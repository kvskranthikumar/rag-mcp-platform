"""Console-script launcher that runs the Streamlit ingestion app.

Uses ``st.App`` so the app can expose a ``/health`` route alongside Streamlit's
own endpoints. Exposed as the ``rag-mcp-platform-ingest-ui`` command. Streamlit config
flags are passed through, e.g.:
    rag-mcp-platform-ingest-ui --server.port 8502 --server.address 0.0.0.0

``STREAMLIT_*`` environment variables are honoured too (CLI flags win). Unlike
``streamlit run``, ``st.App.run()`` does not read those itself, so they are
resolved here from Streamlit's own config metadata.

Port: 8501 locally (Streamlit default). The container image sets
``STREAMLIT_SERVER_PORT=8080``.

Behind a reverse proxy that forwards a path prefix, set the prefix so Streamlit
emits matching asset and WebSocket URLs, e.g.::

    STREAMLIT_SERVER_BASE_URL_PATH=documents/ingestion
    rag-mcp-platform-ingest-ui --server.baseUrlPath documents/ingestion
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _cast(value: str, option_type: type | None) -> Any:
    if option_type is bool:
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        return value
    if option_type in (int, float):
        try:
            return option_type(value)
        except ValueError:
            return value
    if option_type is list:
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


def _parse_flags(args: list[str]) -> list[tuple[str, str]]:
    """``--server.port 8502`` / ``--server.port=8502`` -> ("server.port", "8502")."""
    parsed: list[tuple[str, str]] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("--"):
            i += 1
            continue
        key, sep, inline = arg[2:].partition("=")
        key = key.replace("_", ".")
        if sep:
            parsed.append((key, inline))
            i += 1
        elif i + 1 < len(args) and not args[i + 1].startswith("--"):
            parsed.append((key, args[i + 1]))
            i += 2
        else:
            parsed.append((key, "true"))
            i += 1
    return parsed


def _config_overrides(args: list[str]) -> dict[str, Any]:
    from streamlit import config

    template = config._config_options_template
    overrides: dict[str, Any] = {}

    # Sensitive options are read from the environment by Streamlit itself.
    for name, option in template.items():
        if option.sensitive:
            continue
        env_var = getattr(option, "env_var", None)
        if env_var and env_var in os.environ:
            overrides[name] = _cast(os.environ[env_var], option.type)

    for key, raw in _parse_flags(args):
        option = template.get(key)
        overrides[key] = _cast(raw, option.type if option else None)

    # Streamlit builds its asset and /_stcore/stream URLs from this prefix, so a
    # proxy that forwards the prefix (Azure Application Gateway) needs it set or
    # the WebSocket handshake 404s and the client retries forever.
    base_url_path = str(overrides.get("server.baseUrlPath") or "").strip().strip("/")
    if base_url_path:
        overrides["server.baseUrlPath"] = base_url_path
    else:
        overrides.pop("server.baseUrlPath", None)

    return overrides


def main() -> None:
    import streamlit as st

    from rag_mcp_platform.ingest_ui.health import health_routes

    overrides = _config_overrides(sys.argv[1:])
    app_path = str(Path(__file__).parent / "app.py")
    app = st.App(app_path, routes=health_routes(str(overrides.get("server.baseUrlPath", ""))))
    app.run(config=overrides or None)


if __name__ == "__main__":
    main()
