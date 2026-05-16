-- 002: Channels.
-- One row per company / tenant. `metadata` is a jsonb bag for arbitrary
-- per-channel tags (industry, region, custom keys) so adding a new
-- attribute doesn't need another migration.
CREATE TABLE IF NOT EXISTS channels (
    "Id"           uuid          PRIMARY KEY DEFAULT gen_random_uuid(),
    "Name"         text          NOT NULL UNIQUE,
    "Description"  text          NOT NULL DEFAULT '',
    metadata       jsonb         NOT NULL DEFAULT '{}'::jsonb,
    "CreatedAt"    timestamptz   NOT NULL DEFAULT now()
);
