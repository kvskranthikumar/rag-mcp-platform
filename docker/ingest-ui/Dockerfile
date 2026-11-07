# Streamlit ingestion UI
# Build from the repository root so COPY can see pyproject.toml and src/:
#   docker build -f docker/ingest-ui/Dockerfile -t rag-mcp-platform-ingest-ui .
# From this folder (docker/ingest-ui) you must pass the repo root as context:
#   docker build -f Dockerfile -t rag-mcp-platform-ingest-ui ../..
#   docker run --rm -p 8080:8080 --env-file .env rag-mcp-platform-ingest-ui
#
# Future images follow the same pattern, e.g.:
#   docker/web/Dockerfile   ->  docker build -f docker/web/Dockerfile -t rag-mcp-platform-web .
#   docker/mcp/Dockerfile   ->  docker build -f docker/mcp/Dockerfile -t rag-mcp-platform-mcp .

FROM python:3.12-slim-bookworm

# Set STREAMLIT_SERVER_BASE_URL_PATH at deploy time when a proxy forwards a path
# prefix (e.g. documents/ingestion), otherwise /_stcore/stream 404s.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# pymupdf / lxml wheels are usually sufficient on slim; add build deps only if pip fails.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -e ".[ingest-ui]"

USER 1800
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["rag-mcp-platform-ingest-ui"]
