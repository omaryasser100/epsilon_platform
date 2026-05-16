-- 001: Extensions.
-- pgvector ships built into the pgvector/pgvector:pg16 image; this is
-- here so future installs against a vanilla Postgres also work.
CREATE EXTENSION IF NOT EXISTS vector;
