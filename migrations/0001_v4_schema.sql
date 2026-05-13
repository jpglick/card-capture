-- migrations/0001_v4_schema.sql
-- All statements wrapped in IF NOT EXISTS / try-except for idempotency.

-- 1. truth_files
-- Stores the ground-truth labeling payload for each video (one row per video).
-- Written by: Surface D (labeling endpoints). Read by: Surface D (harness), Surface B (labeling UX).
-- JSON payload: truth.json schema (Contract 4).
CREATE TABLE IF NOT EXISTS truth_files (
    video_id        TEXT    PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    payload_json    TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 2. regression_baselines
-- Named snapshots of a frozen pipeline configuration used as comparison targets.
-- Written by: Surface D (promote-baseline endpoint). Read by: Surface D (harness), Surface B (Regression tab).
-- config_json: the full pipeline config at the time of baseline creation.
CREATE TABLE IF NOT EXISTS regression_baselines (
    baseline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    code_sha    TEXT    NOT NULL,
    config_json TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 3. regression_runs
-- One row per harness execution. Stores aggregate metrics and per-video breakdowns.
-- Written by: Surface D (harness). Read by: Surface D, Surface B (Regression tab).
-- metrics_json: {card_recall, card_precision, side_accuracy, dedup_ari, image_quality_ssim}
-- per_video_json: [{video_id, metrics: {...}, regressions: [...]}]
CREATE TABLE IF NOT EXISTS regression_runs (
    run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_id    INTEGER REFERENCES regression_baselines(baseline_id),
    code_sha       TEXT    NOT NULL,
    config_json    TEXT    NOT NULL,
    metrics_json   TEXT    NOT NULL,
    per_video_json TEXT    NOT NULL,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 4. fb_labels
-- Single front/back/uncertain verdicts produced by the F/B trainer UX.
-- Written by: Surface D (POST /label/fb). Read by: Surface C (training pipeline).
-- labeler: "human" | "model:<model_name>"
CREATE TABLE IF NOT EXISTS fb_labels (
    label_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_run_id INTEGER,
    instance_id   TEXT    NOT NULL,
    frame_index   INTEGER NOT NULL,
    side          TEXT    NOT NULL CHECK (side IN ('front', 'back', 'uncertain')),
    labeler       TEXT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 5. dedup_clusters
-- Groups of card instances predicted (and optionally confirmed) to be the same physical card.
-- Written by: Surface A dedup step (predicted), Surface D cluster-UX (confirmed).
-- Read by: Surface B (dedup cluster UX), Surface C (dedup training).
-- predicted_member_ids_json: ["<instance_id>", ...]
-- confirmed_member_ids_json: null until an operator confirms the cluster.
-- status: "unverified" -> "confirmed" | "split" | "merged"
CREATE TABLE IF NOT EXISTS dedup_clusters (
    cluster_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_member_ids_json TEXT    NOT NULL,
    confirmed_member_ids_json TEXT,
    status                    TEXT    NOT NULL CHECK (status IN ('unverified', 'confirmed', 'split', 'merged')),
    updated_at                TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 6. model_versions
-- Registry of trained model checkpoints with eval metrics.
-- Written by: Surface C (training pipeline). Read by: Surface B (Train tab), Surface A (model loader).
-- eval_metrics_json: {"val_accuracy": 0.94, "val_f1": 0.93, ...}
-- checkpoint_path: absolute or repo-relative path to .pt / .mlpackage file.
CREATE TABLE IF NOT EXISTS model_versions (
    version_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name          TEXT    NOT NULL,
    training_set_hash   TEXT    NOT NULL,
    eval_metrics_json   TEXT    NOT NULL,
    checkpoint_path     TEXT    NOT NULL,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (model_name, training_set_hash)
);

-- 7. hard_cases
-- Auto-captured edge-case frames surfaced in the Hard Cases tab for operator review.
-- Written by: Surface A pipeline (existing hard_case_capture.py integration). Read by: Surface B, Surface C.
-- stage_id: Metaflow step name where the case was captured (e.g. "novelty", "score").
-- reason: short string e.g. "novelty_below_threshold", "low_confidence", "blur"
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

-- Extensions to existing pipeline_events table
-- Applied with try/except at migration runtime (SQLite < 3.35.5 has no IF NOT EXISTS for ADD COLUMN).
ALTER TABLE pipeline_events ADD COLUMN stage_id TEXT;
ALTER TABLE pipeline_events ADD COLUMN artifact_ref TEXT;

-- Indices
CREATE INDEX IF NOT EXISTS idx_regression_runs_baseline ON regression_runs(baseline_id);
CREATE INDEX IF NOT EXISTS idx_fb_labels_instance        ON fb_labels(instance_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_events_stage     ON pipeline_events(stage_id);
