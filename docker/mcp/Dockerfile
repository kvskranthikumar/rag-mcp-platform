# Streamable HTTP MCP server
# Build from the repository root:
#   docker build -f docker/mcp/Dockerfile -t rag-mcp-platform-mcp .
# Run (inject secrets via OS env — no .env file in the image):
#   docker run --rm -p 8080:8080 --env-file .env rag-mcp-platform-mcp
#
# The container sets RAG_MCP_PLATFORM_RUNTIME=container so Settings reads only OS
# environment variables, not a dotenv file baked into the image.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_MCP_PLATFORM_RUNTIME=container \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -e ".[mcp-http]"

USER 1800
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["rag-mcp-platform-mcp-http"]
