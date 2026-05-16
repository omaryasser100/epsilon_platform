-- 000: Schema migration ledger.
--
-- Tracks which numbered files have been applied so the migration
-- runner (scripts/migrate.py) can be idempotent and tell new
-- contributors what state their DB is in.
--
-- Apply this file before any other migration.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text         PRIMARY KEY,
    applied_at  timestamptz  NOT NULL DEFAULT now()
);
