-- RAG pipeline tables — epsilon_rag layer.
-- Runs after init_control.sql (alphabetical order in initdb.d/).
-- Linked to control.channel via control.channel.rag_channel_id = channels."Id".

CREATE EXTENSION IF NOT EXISTS vector;

-- Channels (one per RAG tenant / company)
CREATE TABLE IF NOT EXISTS channels (
    "Id"          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    "Name"        text        NOT NULL UNIQUE,
    "Description" text        NOT NULL DEFAULT '',
    metadata      jsonb       NOT NULL DEFAULT '{}'::jsonb,
    "CreatedAt"   timestamptz NOT NULL DEFAULT now()
);

-- Reports (one per ingested PDF)
CREATE TABLE IF NOT EXISTS reports (
    "Id"        uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    "ChannelId" uuid        NOT NULL REFERENCES channels("Id") ON DELETE CASCADE,
    "Filename"  text        NOT NULL,
    "Title"     text        NOT NULL DEFAULT '',
    metadata    jsonb       NOT NULL DEFAULT '{}'::jsonb,
    "CreatedAt" timestamptz NOT NULL DEFAULT now(),
    UNIQUE ("ChannelId", "Filename")
);

CREATE INDEX IF NOT EXISTS reports_channelid_idx ON reports ("ChannelId");

-- Chunks with bge-m3 hybrid (dense + sparse) vectors
CREATE TABLE IF NOT EXISTS report_chunks (
    "Id"             uuid              PRIMARY KEY DEFAULT gen_random_uuid(),
    "ChannelId"      uuid              NOT NULL REFERENCES channels("Id") ON DELETE CASCADE,
    "ReportId"       uuid              NOT NULL REFERENCES reports("Id")  ON DELETE CASCADE,
    "PageNumber"     integer           NOT NULL,
    "ChunkIndex"     integer           NOT NULL,
    "Content"        text              NOT NULL,
    embedding        vector(1024)      NOT NULL,
    sparse_embedding sparsevec(250002) NOT NULL,
    metadata         jsonb             NOT NULL DEFAULT '{}'::jsonb,
    "CreatedAt"      timestamptz       NOT NULL DEFAULT now(),
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

-- Job lifecycle tracking (ops visibility)
CREATE TABLE IF NOT EXISTS ai_jobs (
    "Id"           uuid        PRIMARY KEY,
    "Status"       text        NOT NULL DEFAULT 'Pending',
    "OutputData"   jsonb,
    "ErrorMessage" text,
    "StartedAt"    timestamptz,
    "CompletedAt"  timestamptz,
    "CreatedAt"    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ai_jobs_status_idx ON ai_jobs ("Status");
