-- 005: ai_jobs — job lifecycle tracking. Kept for ops visibility; the
-- local CLI doesn't use it directly.
CREATE TABLE IF NOT EXISTS ai_jobs (
    "Id"            uuid          PRIMARY KEY,
    "Status"        text          NOT NULL DEFAULT 'Pending',
    "OutputData"    jsonb,
    "ErrorMessage"  text,
    "StartedAt"     timestamptz,
    "CompletedAt"   timestamptz,
    "CreatedAt"     timestamptz   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ai_jobs_status_idx ON ai_jobs ("Status");
