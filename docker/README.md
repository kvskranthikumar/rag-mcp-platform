# Container Images

Build from the repository root so each Dockerfile can copy the package source:

```powershell
docker build -f docker/mcp/Dockerfile -t rag-mcp-platform-mcp .
docker build -f docker/ingest-ui/Dockerfile -t rag-mcp-platform-ingest-ui .
```

Run the MCP HTTP server:

```powershell
docker run --rm -p 8080:8080 --env-file .env rag-mcp-platform-mcp
```

Run the ingestion UI:

```powershell
docker run --rm -p 8080:8080 --env-file .env rag-mcp-platform-ingest-ui
```

Both containers connect to an external PostgreSQL instance and embedding provider. Create the database schema with `rag-mcp-platform init-db` before starting ingestion. Inject secrets through your runtime environment or secret manager; do not copy credentials into an image. Local sample documents are not copied into the containers.

The MCP HTTP endpoint is `/mcp`, with `GET /health` for health checks. The ingestion UI listens on port `8080` in the container. If a reverse proxy forwards a path prefix, set `STREAMLIT_SERVER_BASE_URL_PATH` to the same prefix and preserve WebSocket forwarding.

For a registry, replace the local image tags with your own registry and version tags. This guide intentionally contains no organization-specific subscriptions, registry names, hostnames, or deployment paths.