-- 004: report_chunks — bge-m3 hybrid (dense + sparse) chunk vectors.
-- ChannelId is denormalised from reports.ChannelId so the retrieval
-- HNSW scans can filter without a JOIN.
CREATE TABLE IF NOT EXISTS report_chunks (
    "Id"              uuid              PRIMARY KEY DEFAULT gen_random_uuid(),
    "ChannelId"       uuid              NOT NULL REFERENCES channels("Id") ON DELETE CASCADE,
    "ReportId"        uuid              NOT NULL REFERENCES reports("Id")  ON DELETE CASCADE,
    "PageNumber"      integer           NOT NULL,
    "ChunkIndex"      integer           NOT NULL,
    "Content"         text              NOT NULL,
    embedding         vector(1024)      NOT NULL,
    sparse_embedding  sparsevec(250002) NOT NULL,
    metadata          jsonb             NOT NULL DEFAULT '{}'::jsonb,
    "CreatedAt"       timestamptz       NOT NULL DEFAULT now(),
    UNIQUE ("ReportId", "PageNumber", "ChunkIndex")
);

CREATE INDEX IF NOT EXISTS report_chunks_reportid_idx
    ON report_chunks ("ReportId");
CREATE INDEX IF NOT EXISTS report_chunks_channelid_idx
    ON report_chunks ("ChannelId");
CREATE INDEX IF NOT EXISTS report_chunks_embedding_hnsw_idx
    ON report_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS report_chunks_sparse_embedding_hnsw_idx
    ON report_chunks USING hnsw (sparse_embedding sparsevec_ip_ops);
