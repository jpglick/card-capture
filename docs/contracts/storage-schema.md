# Contract 1 — Storage Schema Additions

**Status:** Frozen (Wave 1 sign-off)  
**Owner:** Surface A (Orchestration / Pipeline)  
**Consumers:** Surface B (reads cards, runs, clusters via API), Surface C (reads/writes `fb_labels`, `model_versions`, `dedup_clusters`), Surface D (writes/reads `truth_files`, `regression_baselines`, `regression_runs`, `hard_cases`)

> **Stability guarantee.** Table names, column names, and column types listed here will not change without explicit four-surface ack. Additive changes (new optional columns, new tables) may be proposed by any surface and require four-surface ack before landing.

---

## Overview

All data lives in a single SQLite database (`cards.sqlite`). Seven new tables are added in `migrations/0001_v4_schema.sql`. Two columns are added to the existing `pipeline_events` table.

Surface A runs all migrations via `migrations/run_migrations.py` at startup. Migrations are idempotent (safe to run repeatedly).

---

## DDL

```sql
-- migrations/0001_v4_schema.sql
-- All statements wrapped in IF NOT EXISTS / try-except for idempotency.

-- 1. truth_files
-- Stores the ground-truth labeling payload for each video (one row per video).
-- Written by: Surface D (labeling endpoints). Read by: Surface D (harness), Surface B (labeling UX).
-- JSON payload: truth.json schema (Contract 4).
CREATE TABLE IF NOT EXISTS truth_files (
    video_id        TEXT    PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    payload_json    TEXT    NOT NULL,            -- JSON blob: see Contract 4
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 2. regression_baselines
-- Named snapshots of a frozen pipeline configuration used as comparison targets.
-- Written by: Surface D (promote-baseline endpoint). Read by: Surface D (harness), Surface B (Regression tab).
-- config_json: the full pipeline config at the time of baseline creation.
CREATE TABLE IF NOT EXISTS regression_baselines (
    baseline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,         -- e.g. "baseline_v4.1"
    code_sha    TEXT    NOT NULL,                -- git commit SHA
    config_json TEXT    NOT NULL,                -- JSON blob: pipeline config snapshot
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 3. regression_runs
-- One row per harness execution. Stores aggregate metrics and per-video breakdowns.
-- Written by: Surface D (harness). Read by: Surface D, Surface B (Regression tab).
-- metrics_json: {card_recall, card_precision, side_accuracy, dedup_ari, image_quality_ssim}
-- per_video_json: [{video_id, metrics: {...}, regressions: [...]}]
CREATE TABLE IF NOT EXISTS regression_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_id   INTEGER REFERENCES regression_baselines(baseline_id),
    code_sha      TEXT    NOT NULL,
    config_json   TEXT    NOT NULL,              -- JSON blob: pipeline config at run time
    metrics_json  TEXT    NOT NULL,              -- JSON blob: aggregate metrics
    per_video_json TEXT   NOT NULL,              -- JSON blob: per-video metric breakdowns
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 4. fb_labels
-- Single front/back/uncertain verdicts produced by the F/B trainer UX.
-- Written by: Surface D (POST /label/fb). Read by: Surface C (training pipeline).
-- labeler: "human" | "model:<model_name>"
CREATE TABLE IF NOT EXISTS fb_labels (
    label_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_run_id INTEGER,                       -- FK to pipeline run (loosely coupled; nullable)
    instance_id   TEXT    NOT NULL,              -- UUID-4 track instance_id
    frame_index   INTEGER NOT NULL,
    side          TEXT    NOT NULL CHECK (side IN ('front', 'back', 'uncertain')),
    labeler       TEXT,                          -- "human" or "model:<name>"
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- 5. dedup_clusters
-- Groups of card instances predicted (and optionally confirmed) to be the same physical card.
-- Written by: Surface A dedup step (predicted), Surface D cluster-UX (confirmed).
-- Read by: Surface B (dedup cluster UX), Surface C (dedup training).
-- predicted_member_ids_json: ["<instance_id>", ...]
-- confirmed_member_ids_json: null until an operator confirms the cluster.
-- status: "unverified" → "confirmed" | "split" | "merged"
CREATE TABLE IF NOT EXISTS dedup_clusters (
    cluster_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_member_ids_json TEXT    NOT NULL,  -- JSON array of instance UUIDs
    confirmed_member_ids_json TEXT,              -- JSON array; NULL until verified
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
    model_name          TEXT    NOT NULL,        -- e.g. "fb_classifier", "presence_classifier"
    training_set_hash   TEXT    NOT NULL,        -- SHA256 of the training dataset manifest
    eval_metrics_json   TEXT    NOT NULL,        -- JSON blob
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
    run_id            TEXT,                      -- Metaflow run ID (loosely coupled; nullable for legacy)
    frame_index       INTEGER,
    stage_id          TEXT    NOT NULL,          -- Metaflow step name
    reason            TEXT    NOT NULL,
    thumbnail_path    TEXT,
    source_frame_path TEXT,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Extensions to existing pipeline_events table
-- Applied with try/except at migration runtime (SQLite < 3.35.5 has no IF NOT EXISTS for ADD COLUMN).
ALTER TABLE pipeline_events ADD COLUMN stage_id TEXT;       -- Metaflow step name
ALTER TABLE pipeline_events ADD COLUMN artifact_ref TEXT;   -- "<run_id>/<step>/<artifact_name>"

-- Indices
CREATE INDEX IF NOT EXISTS idx_regression_runs_baseline ON regression_runs(baseline_id);
CREATE INDEX IF NOT EXISTS idx_fb_labels_instance        ON fb_labels(instance_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_events_stage     ON pipeline_events(stage_id);
```

---

## Table Summary

| Table | Owner (writes) | Consumers (reads) | JSON blobs | Notes |
|---|---|---|---|---|
| `truth_files` | D | D, B | `payload_json` (Contract 4 shape) | One row per video; overwritten on save |
| `regression_baselines` | D | D, B | `config_json` | Name must be unique; `baseline_v4.1` is the Wave 1 seed |
| `regression_runs` | D | D, B | `metrics_json`, `per_video_json` | FK to `regression_baselines`; nullable if no baseline selected |
| `fb_labels` | D | C | — | `labeler = "human"` for UX-sourced labels |
| `dedup_clusters` | A (predicted), D (confirmed) | B, C | `predicted_member_ids_json`, `confirmed_member_ids_json` | Status starts `unverified` |
| `model_versions` | C | A, B | `eval_metrics_json` | Unique constraint prevents duplicate retrain rows |
| `hard_cases` | A | B, C | — | `run_id` is a Metaflow run-id string, not an integer FK |
| `pipeline_events` (ext.) | A | A, B, D | — | `stage_id` and `artifact_ref` are new nullable columns |

---

## Migration Runner Contract

`migrations/run_migrations.py` exposes a single public function:

```python
def apply_migrations(db_path: Path) -> None: ...
```

- Applies every `*.sql` file in `migrations/` in sorted filename order.
- Tracks applied files in an internal `_migrations` table.
- Idempotent: re-running is safe.
- `ALTER TABLE … ADD COLUMN` failures with `duplicate column name` are silently swallowed.
- Any other `sqlite3.OperationalError` is re-raised.

---

## Change Policy

Changes to this contract require explicit written ack from all four surface owners (A, B, C, D) before any code is merged.
