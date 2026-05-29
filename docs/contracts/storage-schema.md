# Contract 1 — Storage Schema

**Status:** Updated for Wave 3 + v5.5  
**Owner:** Surface A (Orchestration / Pipeline)  
**Consumers:** Surface B (reads cards, runs, clusters via API), Surface C (reads/writes `fb_labels`, `model_versions`, `dedup_clusters`), Surface D (writes/reads `truth_files`, `regression_baselines`, `regression_runs`, `hard_cases`)

> **Stability guarantee.** Table names, column names, and column types listed here will not change without explicit four-surface ack. Additive changes (new optional columns, new tables) may be proposed by any surface and require four-surface ack before landing.

---

## Overview

All data lives in a single SQLite database (`cards.sqlite`). Surface A runs all migrations via `migrations/run_migrations.py` at startup. Migrations are idempotent (safe to run repeatedly).

---

## DDL

```sql
-- 0. videos (Base table)
CREATE TABLE IF NOT EXISTS videos (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path   TEXT    NOT NULL,
    file_hash     TEXT    NOT NULL,
    duration_ms   INTEGER NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'processing',
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 1. pipeline_events
CREATE TABLE IF NOT EXISTS pipeline_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      INTEGER REFERENCES videos(id),
    run_id        TEXT,
    stage_id      TEXT,
    frame_index   INTEGER NOT NULL,
    timestamp_ms  INTEGER NOT NULL,
    event_type    TEXT    NOT NULL,
    data_json     TEXT,
    artifact_ref  TEXT,
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. card_instances
CREATE TABLE IF NOT EXISTS card_instances (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id   TEXT    UNIQUE, -- Logical UUID
    video_id      INTEGER NOT NULL REFERENCES videos(id),
    run_id        TEXT,
    track_id      TEXT    NOT NULL,
    session_id    TEXT,
    start_frame   INTEGER,
    end_frame     INTEGER,
    angle         TEXT,
    visual_hash   TEXT,
    fused_image_path TEXT,
    reid_embedding BLOB,
    is_hidden     INTEGER DEFAULT 0,
    is_duplicate_of INTEGER REFERENCES card_instances(id),
    hidden        INTEGER NOT NULL DEFAULT 0,
    front_crop    TEXT,
    back_crop     TEXT,
    updated_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, track_id)
);

-- 3. card_views
CREATE TABLE IF NOT EXISTS card_views (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    card_instance_id INTEGER NOT NULL REFERENCES card_instances(id),
    instance_id   TEXT    REFERENCES card_instances(instance_id),
    frame_index   INTEGER NOT NULL,
    timestamp_ms  INTEGER NOT NULL,
    image_path    TEXT,
    rectified_path TEXT,
    corners_json  TEXT,
    confidence    REAL,
    quality_score FLOAT,
    quality_score_json TEXT,
    is_canonical  INTEGER NOT NULL DEFAULT 0,
    side          TEXT,
    phash         TEXT,
    reid_embedding BLOB,
    initial_confidence REAL,
    glare_x       REAL,
    glare_y       REAL,
    sharpness     REAL,
    glare_mask_b64 TEXT,
    laplacian_heatmap_b64 TEXT,
    metadata_json TEXT,
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 4. saved_cards (Backward-compatibility)
CREATE TABLE IF NOT EXISTS saved_cards (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id  INTEGER NOT NULL,
    video_id      INTEGER NOT NULL REFERENCES videos(id),
    image_path    TEXT    NOT NULL,
    final_score   REAL    NOT NULL,
    review_state  TEXT    NOT NULL DEFAULT 'pending',
    source_path   TEXT    NOT NULL,
    timestamp_ms  INTEGER NOT NULL,
    score_components_json TEXT NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 5. truth_files
-- Stores the ground-truth labeling payload for each video (one row per video).
-- JSON payload: truth.json schema (Contract 4).
CREATE TABLE IF NOT EXISTS truth_files (
    video_id        TEXT    PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    payload_json    TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    created_at      TEXT    NOT NULL DEFAULT '1970-01-01 00:00:00'
);

-- 6. regression_baselines
-- Named snapshots of a frozen pipeline configuration used as comparison targets.
CREATE TABLE IF NOT EXISTS regression_baselines (
    baseline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    code_sha    TEXT    NOT NULL,
    config_json TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 7. regression_runs
-- One row per harness execution. Stores aggregate metrics and per-video breakdowns.
CREATE TABLE IF NOT EXISTS regression_runs (
    run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_id    INTEGER REFERENCES regression_baselines(baseline_id),
    code_sha       TEXT    NOT NULL,
    config_json    TEXT    NOT NULL,
    metrics_json   TEXT    NOT NULL,
    per_video_json TEXT    NOT NULL,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 8. fb_labels
-- Single front/back/uncertain/no_card verdicts produced by the F/B trainer UX.
CREATE TABLE IF NOT EXISTS fb_labels (
    label_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_run_id INTEGER,
    instance_id   TEXT    NOT NULL,
    frame_index   INTEGER NOT NULL,
    side          TEXT    NOT NULL CHECK (side IN ('front', 'back', 'uncertain', 'no_card')),
    labeler       TEXT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 9. dedup_clusters
-- Groups of card instances predicted (and optionally confirmed) to be the same physical card.
CREATE TABLE IF NOT EXISTS dedup_clusters (
    cluster_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_member_ids_json TEXT    NOT NULL,
    confirmed_member_ids_json TEXT,
    status                    TEXT    NOT NULL CHECK (status IN ('unverified', 'confirmed', 'split', 'merged')),
    updated_at                TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 10. model_versions
-- Registry of trained model checkpoints with eval metrics.
CREATE TABLE IF NOT EXISTS model_versions (
    version_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name          TEXT    NOT NULL,
    training_set_hash   TEXT    NOT NULL,
    eval_metrics_json   TEXT    NOT NULL,
    checkpoint_path     TEXT    NOT NULL,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (model_name, training_set_hash)
);

-- 11. hard_cases
-- Auto-captured edge-case frames surfaced in the Hard Cases tab for operator review.
CREATE TABLE IF NOT EXISTS hard_cases (
    case_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    frame_index       INTEGER,
    stage_id          TEXT    NOT NULL,
    reason            TEXT    NOT NULL,
    thumbnail_path    TEXT,
    source_frame_path TEXT,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 12. config_presets
CREATE TABLE IF NOT EXISTS config_presets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    preset_name TEXT    NOT NULL UNIQUE,
    description TEXT    NOT NULL DEFAULT '',
    config_json TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 13. pipeline_runs
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id      TEXT    PRIMARY KEY,
    video_id    INTEGER NOT NULL REFERENCES videos(id),
    status      TEXT    NOT NULL DEFAULT 'running',
    cards_extracted INTEGER NOT NULL DEFAULT 0,
    started_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    detect_telemetry_json TEXT,
    host_info_json TEXT
);

-- 14. presence_samples
CREATE TABLE IF NOT EXISTS presence_samples (
    id           INTEGER PRIMARY KEY,
    run_id       TEXT    NOT NULL,
    video_id     INTEGER NOT NULL,
    frame_index  INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    image_path   TEXT    NOT NULL,
    label        TEXT    CHECK (label IN ('present', 'absent')),
    labeled_at   TEXT,
    created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
);

-- 15. corner_samples
CREATE TABLE IF NOT EXISTS corner_samples (
    id                INTEGER PRIMARY KEY,
    run_id            TEXT    NOT NULL,
    video_id          INTEGER NOT NULL,
    frame_index       INTEGER NOT NULL,
    image_path        TEXT    NOT NULL,
    predicted_corners TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    label             TEXT    CHECK (label IN ('correct', 'adjusted', 'negative')),
    corrected_corners TEXT,
    labeled_at        TEXT,
    created_at        TEXT    DEFAULT CURRENT_TIMESTAMP
);

-- 16. benchmark_snapshots
CREATE TABLE IF NOT EXISTS benchmark_snapshots (
    id              INTEGER PRIMARY KEY,
    job_id          TEXT    NOT NULL,
    run_id          TEXT    NOT NULL,
    cards_extracted INTEGER NOT NULL,
    snapshotted_at  TEXT    DEFAULT CURRENT_TIMESTAMP
);

-- 17. pipeline_run_logs
CREATE TABLE IF NOT EXISTS pipeline_run_logs (
    id        INTEGER PRIMARY KEY,
    run_id    TEXT    NOT NULL,
    line      TEXT    NOT NULL,
    logged_at TEXT    DEFAULT CURRENT_TIMESTAMP
);

-- 18. run_resource_samples
CREATE TABLE IF NOT EXISTS run_resource_samples (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT    NOT NULL,
    elapsed_s    REAL    NOT NULL,
    cpu_pct      REAL,
    mem_used_mb  REAL,
    mem_pct      REAL,
    gpu_pct      REAL,
    vram_used_mb REAL,
    stage        TEXT    DEFAULT 'init',
    decoder_pct  REAL,
    encoder_pct  REAL,
    mem_io_pct   REAL
);

-- 19. card_view_metrics
CREATE TABLE IF NOT EXISTS card_view_metrics (
    card_instance_id TEXT NOT NULL,
    metric           TEXT NOT NULL,
    value            REAL NOT NULL,
    PRIMARY KEY (card_instance_id, metric)
);

-- 20. telemetry_events
CREATE TABLE IF NOT EXISTS telemetry_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT,
    kind      TEXT NOT NULL,
    payload   TEXT,
    at_ms     INTEGER NOT NULL
);

-- 21. batch_jobs
CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id   TEXT PRIMARY KEY,
    status     TEXT NOT NULL DEFAULT 'queued',
    total      INTEGER NOT NULL DEFAULT 0,
    completed  INTEGER NOT NULL DEFAULT 0,
    failed     INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

---

## Table Summary

| Table | Owner (writes) | Consumers (reads) | Notes |
|---|---|---|---|
| `videos` | A | A, B | Source video metadata |
| `pipeline_events` | A | A, B, D | Fine-grained telemetry and event log |
| `card_instances` | A | B, C | Logical card tracks across frames |
| `card_views` | A | B | Per-frame card crops and metrics |
| `saved_cards` | A | B | Legacy flattened card storage |
| `truth_files` | D | D, B | Ground-truth labels for evaluation |
| `regression_baselines`| D | D, B | Target snapshots for regression testing |
| `regression_runs` | D | D, B | Results of harness executions |
| `fb_labels` | D | C | Training labels for Front/Back classifier |
| `dedup_clusters` | A, D | B, C | Grouping of physical card identity |
| `model_versions` | C | A, B | Registry of ML model weights |
| `hard_cases` | A | B, C | Edge cases flagged for review |
| `config_presets` | B | A | Named sets of pipeline parameters |
| `pipeline_runs` | A | B | Execution metadata for every run |
| `presence_samples` | A, D | C | Dataset for Presence classifier training |
| `corner_samples` | A, D | C | Dataset for Corner refinement training |
| `benchmark_snapshots` | C | B | Point-in-time stats during training |
| `pipeline_run_logs` | A | B | Stdout/stderr capture from runs |
| `run_resource_samples`| A | B | CPU/GPU/Mem utilization over time |
| `card_view_metrics` | A | B | Aggregated metrics for v5.5 DAL |
| `telemetry_events` | A | B | Structured telemetry for v5.5 DAL |
| `batch_jobs` | A | B | Status of cloud batch processing |
