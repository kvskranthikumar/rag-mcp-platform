-- ===========================================================================
-- RAG MCP Platform schema (PostgreSQL + pgvector)
-- ---------------------------------------------------------------------------
-- Run against your existing database, for example:
--   psql "postgresql://user:pass@host:5432/dbname" -f sql/schema.sql
-- or paste into pgAdmin / DBeaver / any SQL client.
--
-- IMPORTANT - embedding dimension:
--   The vector(1536) column below is used with text-embedding-3-large requested
--   at 1536 dims (EMBEDDING_DIM=1536). We use 1536 because pgvector's HNSW index
--   supports at most 2000 dimensions. text-embedding-3-large can emit reduced
--   dimensions via the API `dimensions` parameter with negligible quality loss.
--     text-embedding-3-large @ 1536 (recommended, HNSW-compatible)
--     text-embedding-3-small    -> 1536
--   To keep the FULL 3072 dims, see the two alternatives at the bottom of this
--   file (halfvec index, or exact search with no ANN index).
--   Vectors from different models/dims are not comparable; changing later
--   requires DROP (sql/drop.sql) + re-create + a full re-ingest.
-- ===========================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- --- One row per source document ------------------------------------------
-- embedding_signature records provider:model:dim[:version] used to embed this
-- document, so ingestion re-embeds it when the embedding space changes.
CREATE TABLE IF NOT EXISTS documents (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path         text NOT NULL,
    url                 text,
    title               text,
    doc_type            text NOT NULL,
    content_hash        text NOT NULL,
    embedding_signature text,
    version             int  NOT NULL DEFAULT 1,
    metadata            jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS documents_source_path_key
    ON documents (source_path);

-- Speeds up global content de-duplication (ignore identical content by hash).
CREATE INDEX IF NOT EXISTS documents_content_hash_idx
    ON documents (content_hash);

-- --- One row per chunk (dense vector + generated full-text vector) ----------
CREATE TABLE IF NOT EXISTS chunks (
    id              bigserial PRIMARY KEY,
    document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     int  NOT NULL,
    heading         text,
    content         text NOT NULL,
    token_count     int,
    url_anchor      text,
    embedding       vector(1536),   -- <-- must equal EMBEDDING_DIM (<= 2000 for HNSW)
    embedding_model text,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Dense ANN index (cosine) and lexical (full-text) index for hybrid search.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
-- Tip: raise recall/quality at query time with:  SET hnsw.ef_search = 100;

CREATE INDEX IF NOT EXISTS chunks_tsv_gin
    ON chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS chunks_content_md5_idx ON chunks (md5(content));

-- --- Ingestion run log (observability) -------------------------------------
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

ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS files_failed int NOT NULL DEFAULT 0;
ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS results jsonb NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS trace_id text;

-- ===========================================================================
-- Hybrid search: dense (pgvector cosine) + lexical (BM25-style full-text)
-- fused with Reciprocal Rank Fusion (RRF).
-- ===========================================================================
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

-- ===========================================================================
-- OPTIONAL: keeping the FULL 3072 dimensions of text-embedding-3-large
-- ---------------------------------------------------------------------------
-- The recommended setup above uses vector(1536) so an HNSW index is possible.
-- If you want the native 3072 dims, set EMBEDDING_DIM=3072 and use ONE of these
-- instead of the vector(1536) column + HNSW index above:
--
-- Alternative A - halfvec HNSW index (requires pgvector >= 0.7.0).
--   half-precision vectors support up to 4000 dims for indexing.
--     ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(3072);
--     CREATE INDEX chunks_embedding_hnsw
--         ON chunks USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)
--         WITH (m = 16, ef_construction = 64);
--   NOTE: the search would then need to cast the query, e.g.
--     ORDER BY c.embedding::halfvec(3072) <=> query_embedding::halfvec(3072)
--   (ask the app maintainer to switch the provider/search to halfvec).
--
-- Alternative B - no ANN index (exact search), works on any pgvector version
-- and any dimension. Fine for small/medium corpora (tens of thousands of
-- chunks). Simply use vector(3072) and DO NOT create chunks_embedding_hnsw.
-- Queries stay correct via search_hybrid(); they just scan sequentially.
-- ===========================================================================
