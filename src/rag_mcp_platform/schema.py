"""Database schema management (pgvector).

The embedding dimension is configuration-driven, so the schema is generated from
``settings.embedding_dim``. Model or dimension changes require initializing a
separate database and rebuilding the complete index before switching clients.
"""

from __future__ import annotations

from .db import raw_connect
from .settings import get_settings

HYBRID_SEARCH_FUNCTION = """
CREATE OR REPLACE FUNCTION search_hybrid(
    query_text       text,
    query_embedding  vector,
    match_count      int DEFAULT 10,
    rrf_k            int DEFAULT 60,
    candidate_count  int DEFAULT 50
)
RETURNS TABLE (
    chunk_id     bigint,
    document_id  uuid,
    content      text,
    heading      text,
    url_anchor   text,
    title        text,
    source_path  text,
    url          text,
    metadata     jsonb,
    score        double precision
)
LANGUAGE sql STABLE
AS $$
WITH vector_search AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_embedding) AS rank
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT candidate_count
),
lexical_search AS (
    SELECT c.id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(c.tsv, plainto_tsquery('english', query_text)) DESC
           ) AS rank
    FROM chunks c
    WHERE c.tsv @@ plainto_tsquery('english', query_text)
    ORDER BY ts_rank_cd(c.tsv, plainto_tsquery('english', query_text)) DESC
    LIMIT candidate_count
),
fused AS (
    SELECT COALESCE(v.id, l.id) AS id,
           COALESCE(1.0 / (rrf_k + v.rank), 0.0)
         + COALESCE(1.0 / (rrf_k + l.rank), 0.0) AS score
    FROM vector_search v
    FULL OUTER JOIN lexical_search l ON v.id = l.id
)
SELECT c.id,
       c.document_id,
       c.content,
       c.heading,
       c.url_anchor,
       d.title,
       d.source_path,
       d.url,
       c.metadata,
       f.score
FROM fused f
JOIN chunks c    ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.score DESC
LIMIT match_count;
$$;
"""


def _ddl_statements(dim: int) -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "CREATE EXTENSION IF NOT EXISTS pgcrypto;",
        """
        CREATE TABLE IF NOT EXISTS documents (
            id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            source_path        text NOT NULL,
            url                text,
            title              text,
            doc_type           text NOT NULL,
            content_hash       text NOT NULL,
            embedding_signature text,
            version            int  NOT NULL DEFAULT 1,
            metadata           jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at         timestamptz NOT NULL DEFAULT now(),
            updated_at         timestamptz NOT NULL DEFAULT now()
        );
        """,
        # Idempotent upgrade for databases created before this column existed.
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_signature text;",
        "CREATE UNIQUE INDEX IF NOT EXISTS documents_source_path_key ON documents (source_path);",
        # Speeds up global content de-duplication lookups.
        "CREATE INDEX IF NOT EXISTS documents_content_hash_idx ON documents (content_hash);",
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id              bigserial PRIMARY KEY,
            document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index     int  NOT NULL,
            heading         text,
            content         text NOT NULL,
            token_count     int,
            url_anchor      text,
            embedding       vector({dim}),
            embedding_model text,
            tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            metadata        jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            created_at      timestamptz NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_index)
        );
        """,
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);",
        "CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);",
        "CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);",
        "CREATE INDEX IF NOT EXISTS chunks_content_md5_idx ON chunks (md5(content));",
        """
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            id             bigserial PRIMARY KEY,
            started_at     timestamptz NOT NULL DEFAULT now(),
            finished_at    timestamptz,
            files_seen     int NOT NULL DEFAULT 0,
            files_changed  int NOT NULL DEFAULT 0,
            chunks_written int NOT NULL DEFAULT 0,
            status         text NOT NULL DEFAULT 'running',
            error          text
        );
        """,
        "ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS files_failed "
        "int NOT NULL DEFAULT 0;",
        "ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS results "
        "jsonb NOT NULL DEFAULT '[]'::jsonb;",
        "ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS trace_id text;",
    ]


def init_db(drop: bool = False) -> None:
    settings = get_settings()
    with raw_connect() as conn:
        if drop:
            conn.execute("DROP TABLE IF EXISTS chunks CASCADE;")
            conn.execute("DROP TABLE IF EXISTS documents CASCADE;")
            conn.execute("DROP TABLE IF EXISTS ingestion_runs CASCADE;")
        for stmt in _ddl_statements(settings.embedding_dim):
            conn.execute(stmt)
        conn.execute(HYBRID_SEARCH_FUNCTION)
