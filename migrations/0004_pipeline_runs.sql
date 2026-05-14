-- migrations/0004_pipeline_runs.sql
-- Track every pipeline run (even ones that produce 0 cards).

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id      TEXT    PRIMARY KEY,
    video_id    INTEGER NOT NULL REFERENCES videos(id),
    status      TEXT    NOT NULL DEFAULT 'running',  -- running | completed | failed
    cards_extracted INTEGER NOT NULL DEFAULT 0,
    started_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);
