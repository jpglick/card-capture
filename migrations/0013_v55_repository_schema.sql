-- migrations/0013_v55_repository_schema.sql
-- Add the v5.5-shaped tables required by the data-access-layer repositories.
-- All statements are IF NOT EXISTS so this migration is safe to re-apply.

-- Card-view metrics, one row per (instance, metric).
-- Distinct from the existing `card_views` table (which holds per-frame views);
-- name this `card_view_metrics` to avoid collision.
CREATE TABLE IF NOT EXISTS card_view_metrics (
    card_instance_id TEXT NOT NULL,
    metric           TEXT NOT NULL,
    value            REAL NOT NULL,
    PRIMARY KEY (card_instance_id, metric)
);

-- Card-instance v5.5 surface used by repositories (card_instance_id string,
-- front/back crop paths). The legacy `card_instances` table keeps its integer
-- id; we add columns that may be missing.
-- Note: SQLite < 3.35.5 doesn't support IF NOT EXISTS for ADD COLUMN.
-- The run_migrations.py script handles 'duplicate column' errors.
ALTER TABLE card_instances ADD COLUMN front_crop TEXT;
ALTER TABLE card_instances ADD COLUMN back_crop TEXT;

-- Telemetry events recorded by TelemetryRepository.
CREATE TABLE IF NOT EXISTS telemetry_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT,
    kind      TEXT NOT NULL,
    payload   TEXT,
    at_ms     INTEGER NOT NULL
);

-- Batch jobs for cloud processing.
CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id   TEXT PRIMARY KEY,
    status     TEXT NOT NULL DEFAULT 'queued',
    total      INTEGER NOT NULL DEFAULT 0,
    completed  INTEGER NOT NULL DEFAULT 0,
    failed     INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Ensure fb_labels and truth_files match what repositories expect.
-- They are created in 0001_v4_schema.sql, but we ensure columns here.
-- (ADD COLUMN is idempotent via run_migrations.py tolerance).
ALTER TABLE fb_labels ADD COLUMN label_id INTEGER; -- Might already exist as PK
ALTER TABLE truth_files ADD COLUMN created_at TEXT NOT NULL DEFAULT '1970-01-01 00:00:00';
