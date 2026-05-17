CREATE TABLE IF NOT EXISTS pipeline_run_logs (
    id        INTEGER PRIMARY KEY,
    run_id    TEXT    NOT NULL,
    line      TEXT    NOT NULL,
    logged_at TEXT    DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_run_logs_run_id ON pipeline_run_logs(run_id);
