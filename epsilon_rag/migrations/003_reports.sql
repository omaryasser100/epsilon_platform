-- 003: Reports (PDFs).
-- One row per PDF. (ChannelId, Filename) is the natural key — re-ingesting
-- the same filename into the same channel re-uses this row, and Stage 4 of
-- the ingest pipeline deletes the old report_chunks before re-inserting.
CREATE TABLE IF NOT EXISTS reports (
    "Id"           uuid          PRIMARY KEY DEFAULT gen_random_uuid(),
    "ChannelId"    uuid          NOT NULL REFERENCES channels("Id") ON DELETE CASCADE,
    "Filename"     text          NOT NULL,
    "Title"        text          NOT NULL DEFAULT '',
    metadata       jsonb         NOT NULL DEFAULT '{}'::jsonb,
    "CreatedAt"    timestamptz   NOT NULL DEFAULT now(),
    UNIQUE ("ChannelId", "Filename")
);

CREATE INDEX IF NOT EXISTS reports_channelid_idx ON reports ("ChannelId");
