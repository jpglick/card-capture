-- migrations/0010_resource_samples.sql
ALTER TABLE pipeline_runs ADD COLUMN host_info_json TEXT;

CREATE TABLE IF NOT EXISTS run_resource_samples (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT    NOT NULL,
    elapsed_s    REAL    NOT NULL,
    cpu_pct      REAL,
    mem_used_mb  REAL,
    mem_pct      REAL,
    gpu_pct      REAL,
    vram_used_mb REAL
);
CREATE INDEX IF NOT EXISTS idx_rrs_run_id ON run_resource_samples(run_id);
