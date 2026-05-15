# Training Hub Design Spec
**Date:** 2026-05-14  
**Status:** Approved

---

## Goal

Replace the current scattered labeling pages (`/label/fb`, `/label/hard_cases`, etc.)
with a single unified `/training` route that handles all labeling tasks, retraining,
and benchmarking with minimal steps. The user uploads videos, the pipeline runs, and
frames appear in the labeling queue automatically. All interactions are keyboard-driven.

---

## Context

Three ML models need labeled training data:

| Model | File | What it decides |
|---|---|---|
| Presence classifier | `models/presence_classifier.pt` | Is there a card in this 192px scan frame? |
| Front/Back classifier | `models/fb_classifier.pt` | Is this the front or back of a card? |
| YOLO-OBB detector | `models/yolo_corners.pt` | Where are the card corners in this frame? |

Existing infrastructure to reuse:
- `TrainingService` — in-process job queue for retrain jobs, already wired to `/api/v1/training`
- `fb_labels` table — stores front/back human labels
- `regression_baselines` / `regression_runs` tables — before/after pipeline comparison
- `/api/v1/training/retrain/{model_name}` — already works for `fb_classifier`

---

## Page Structure

Single route: `/training`

```
┌─────────────────────────────────────────────────────────┐
│  TRAINING                                               │
│                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  Presence    │ │  Front/Back  │ │ YOLO Corners │   │
│  │  47 pending  │ │  12 pending  │ │   8 pending  │   │
│  │  acc: 71%    │ │  acc: 94%    │ │  acc: —      │   │
│  │ [Label now]  │ │ [Label now]  │ │ [Label now]  │   │
│  └──────────────┘ └──────────────┘ └──────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  BENCHMARK                                       │  │
│  │  Last retrain: 2 days ago                        │  │
│  │  Presence: 71% → 84%  FB: 94% → 94%             │  │
│  │  [Retrain all]  [Run pipeline on last 3 videos]  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  Label history / accuracy charts (below fold)          │
└─────────────────────────────────────────────────────────┘
```

Clicking "Label now" on any panel navigates to that model's labeling flow.
Back button returns to the hub.

---

## Labeling Flows

All three flows share the same chrome: counter top-right, progress bar bottom,
keyboard shortcuts always visible, Skip available on every frame.

### Presence

Classifies 192px scan frames extracted from the pipeline's fast-scan pass.

**Display:** Upscaled scan frame (bilinear, 3–4× for readability), no other UI chrome.

**Options:**

| Key | Action | Training effect |
|---|---|---|
| Y | Card present | Positive example |
| N | No card | Negative example |
| S | Skip | Not stored |

### Front / Back

Classifies rectified 750×1050 card crops produced by the pipeline.

**Display:** Full-quality card crop, centered. Showing the actual pipeline output
lets the user catch bad warps at the same time as labeling.

**Options:**

| Key | Action | Training effect |
|---|---|---|
| F | Front | Positive front example |
| B | Back | Positive back example |
| U | Uncertain (card present, side unclear) | Excluded from training |
| X | Not a card (pipeline false positive) | Negative example for presence + YOLO |
| S | Skip | Not stored |

### YOLO Corners

Verifies or corrects the corner polygon YOLO predicted on a full-resolution frame.
Only borderline detections (confidence 0.50–0.70) are queued — these are the highest-value
training examples.

**Display:** Full frame at display scale. YOLO's predicted corners overlaid as a green
polygon. Confidence badge shown so the user knows how uncertain the model was.

**Options:**

| Key | Action | Training effect |
|---|---|---|
| Y | Corners correct | Positive example with existing corners |
| E | Adjust (enters drag mode) | Positive example with corrected corners |
| N | No card here | Hard negative example |
| S | Skip | Not stored |

In adjust mode: four corner handles become draggable. Spacebar confirms. Escape cancels
and returns to the overlay view.

---

## Auto-Sampling (Queue Population)

After every pipeline run completes, a background task automatically samples frames
into each queue. No user action required.

| Queue | Source | Volume | Balance rule |
|---|---|---|---|
| Presence | Frames from the fast-scan pass — both card-present windows and inter-window gap frames | 20 per run | If queue has >3× more positives than negatives, sample only negatives next run (and vice versa) |
| Front/Back | Every saved card instance produced by the run | All | N/A — all instances are valuable |
| YOLO | Detections where YOLO confidence was 0.50–0.70 | All borderline | N/A |

Sampling writes to new DB tables (see Data Model below). The pipeline run's `run_id`
is stored with each sample so labels can be traced back to their source video.

---

## Retraining

**Trigger:** "Retrain all" button or per-model "Retrain" button on the hub.

**Behavior:**
1. Calls `TrainingService.start_retrain(model_name)` for each model with new labels
   since the last retrain.
2. Progress shown inline on the benchmark panel (epoch X/N, live accuracy).
3. On completion:
   - Accuracy scores update immediately from a held-out 20% split of labeled frames.
     The split is deterministic (label ID % 5 == 0 = held-out) so accuracy numbers
     are comparable across retrains — the validation set does not change between runs.
   - Accuracy is stored in `model_versions.eval_metrics_json` (existing column).
   - A baseline snapshot is written to `regression_baselines` so the next benchmark
     comparison has a reference point.
4. New model weights are written to `models/<model_name>.pt`, replacing the previous
   weights. The old weights are archived to `models/archive/<model_name>_<timestamp>.pt`.

**Accuracy display** (shown immediately, no video re-run needed):

```
  Presence      71%  →  84%   ↑ +13pp
  Front/Back    94%  →  94%   → no change
  YOLO corners  —   →  88%   new model
```

---

## Pipeline Benchmark

**Trigger:** "Run pipeline on last N videos" button (N defaults to 3, configurable in
Settings). If fewer than N videos have been processed, re-runs all available videos
without error.

**Behavior:**
1. Fetches the N most recently processed videos by `started_at` from `pipeline_runs`.
2. Re-runs the full pipeline on each with the new model weights.
3. Compares output against the baseline snapshot saved at last retrain time.
4. Displays a before/after table:

```
  Video              Before        After         Δ
  IMG_5595.MOV       0 cards       4 cards      +4  ✓
  IMG_5596.MOV       0 cards       6 cards      +6  ✓
  IMG_5872.MOV       7 cards       7 cards       0  →

  Avg quality score  0.71          0.76         +7%
```

The benchmark runs asynchronously. Progress is shown inline on the benchmark panel
using the existing SSE event bus.

---

## Stats Area (Below Fold)

- Accuracy over time line chart per model (one series per model, X axis = retrain date)
- Labels added this week bar chart (Presence / Front/Back / YOLO)
- Total labeled count and unlabeled queue depth

---

## Data Model

### New tables

```sql
-- Presence labeling queue and labels
CREATE TABLE presence_samples (
    id          INTEGER PRIMARY KEY,
    run_id      TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    image_path  TEXT NOT NULL,      -- path to saved 192px scan frame
    label       TEXT,               -- 'present' | 'absent' | NULL (unlabeled)
    labeled_at  TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

-- YOLO corner labeling queue and labels
CREATE TABLE corner_samples (
    id              INTEGER PRIMARY KEY,
    run_id          TEXT NOT NULL,
    frame_index     INTEGER NOT NULL,
    image_path      TEXT NOT NULL,  -- full-resolution frame path
    predicted_corners TEXT NOT NULL, -- JSON array of 4 [x,y] pairs
    confidence      REAL NOT NULL,
    label           TEXT,           -- 'correct' | 'adjusted' | 'negative' | NULL
    corrected_corners TEXT,         -- JSON array if adjusted
    labeled_at      TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);
```

The existing `fb_labels` table covers Front/Back labels. No schema change needed there.

### No new columns needed on existing tables

`model_versions.eval_metrics_json` (existing) stores held-out accuracy as
`{"accuracy": 0.84, "val_loss": 0.21}`. No schema migration required.

---

## API Additions

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/training/presence/next` | Next unlabeled presence sample |
| POST | `/api/v1/training/presence/label` | Submit presence label |
| GET | `/api/v1/training/corners/next` | Next unlabeled corner sample |
| POST | `/api/v1/training/corners/label` | Submit corner label (correct/adjusted/negative) |
| GET | `/api/v1/training/stats` | Hub stats (pending counts, accuracy, history) |
| POST | `/api/v1/training/benchmark` | Trigger pipeline benchmark on last N videos |
| GET | `/api/v1/training/benchmark/{job_id}` | Benchmark job status and results |

The existing `/api/v1/training/retrain/{model_name}` and `/api/v1/label/fb/*` endpoints
are unchanged.

---

## Frontend Components

| Component | Route / Location | Purpose |
|---|---|---|
| `TrainingHub` | `/training` | Four-panel hub, stats area |
| `PresenceLabeler` | `/training/presence` | Yes/No scan frame flow |
| `FrontBackLabeler` | `/training/fb` | Replaces `/label/fb` |
| `CornerLabeler` | `/training/corners` | Overlay + drag-to-adjust flow |
| `BenchmarkPanel` | Embedded in hub | Retrain buttons, accuracy diff, pipeline benchmark |
| `AccuracyChart` | Embedded in hub | Accuracy over time per model |

`FrontBackLabeler` replaces the existing `/label/fb` page. The old route redirects.

---

## Prerequisites

`TrainingService.start_retrain()` is currently a stub for `fb_classifier` ("Real training
would happen here"). The implementation plan must include wiring real training scripts
for both `presence` and `fb_classifier` before the Retrain buttons are functional.
`models/fb_classifier.pt` does not yet exist — the first retrain creates it.

---

## Out of Scope

- Active learning / uncertainty sampling (auto-queue from confidence scores) — deferred
- Labeling by multiple users / labeler attribution — deferred  
- Export of labeled datasets to external formats — deferred
- YOLO fine-tuning pipeline (training script) — separate task; this spec covers data
  collection and UI only. Presence and FB retraining already work via `TrainingService`.
