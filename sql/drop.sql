-- ===========================================================================
-- Drop all RAG MCP Platform objects. Use before recreating the schema when changing
-- the embedding model/dimension (followed by sql/schema.sql + a full re-ingest).
--   psql "postgresql://user:pass@host:5432/dbname" -f sql/drop.sql
-- ===========================================================================

DROP FUNCTION IF EXISTS search_hybrid(text, vector, int, int, int);
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS ingestion_runs CASCADE;
