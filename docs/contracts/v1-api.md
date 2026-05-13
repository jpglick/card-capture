# Contract 2 — Service-Layer API (v1)

**Status:** Frozen (Wave 1 sign-off)  
**Owner:** Surface A (Orchestration / Pipeline)  
**Consumers:** Surface B (frontend), Surface D (harness + labeling endpoints)

> **Stability guarantee.** Route paths, HTTP verbs, required request fields, and required response fields listed here will not change without explicit four-surface ack. Additive response fields (new optional JSON keys) and new routes may be proposed and require four-surface ack before landing.

---

## Base URL

All REST routes are prefixed: **`/api/v1`**

SSE channel: **`/events/{run_id}`** (no prefix)

All responses use `Content-Type: application/json` unless noted. All timestamps are ISO-8601 strings (UTC). All IDs are strings unless noted as `integer`.

---

## Error Shape

Every non-2xx response uses:

```json
{
  "detail": "human-readable error message"
}
```

Standard codes: `400` (bad request), `404` (not found), `409` (conflict), `422` (validation error), `501` (not yet implemented — used by stub routes).

---

## 1. Videos

### `GET /api/v1/videos`

List all ingested videos.

**Response `200`:**
```json
[
  {
    "video_id": "practice_session_03",
    "filename": "practice_session_03.mov",
    "duration_ms": 312000,
    "status": "completed",
    "created_at": "2026-05-12T14:00:00Z"
  }
]
```

**Fields:**
| Field | Type | Notes |
|---|---|---|
| `video_id` | string | Stable identifier (stem of filename) |
| `filename` | string | Original filename |
| `duration_ms` | integer | Video duration in milliseconds |
| `status` | string | `"pending"` \| `"processing"` \| `"completed"` \| `"failed"` |
| `created_at` | string | ISO-8601 timestamp |

---

### `POST /api/v1/videos`

Register a video file for processing (multipart upload or path registration).

**Request body** (`multipart/form-data` or `application/json`):
```json
{
  "filename": "practice_session_03.mov",
  "file_path": "/data/videos/practice_session_03.mov"
}
```

**Response `201`:** Same shape as a single Video object above.

**Errors:** `409` if `video_id` already exists.

---

### `GET /api/v1/videos/{video_id}`

Get a single video's metadata.

**Response `200`:** Single Video object (same shape as list item).

**Errors:** `404` if not found.

---

### `DELETE /api/v1/videos/{video_id}`

Remove a video registration (does not delete the source file).

**Response `204`:** No body.

**Errors:** `404` if not found.

---

### `POST /api/v1/videos/{video_id}/process`

Enqueue a pipeline run for the video.

**Request body:**
```json
{
  "config_preset": "balanced",
  "detector": "docaligner",
  "output_dir": "/tmp/out"
}
```

**Fields:**
| Field | Type | Required | Notes |
|---|---|---|---|
| `config_preset` | string | no | `"fast"` \| `"balanced"` \| `"quality"` (default: `"balanced"`) |
| `detector` | string | no | `"docaligner"` \| `"fake"` (default: `"docaligner"`) |
| `output_dir` | string | no | Absolute path; server picks default if omitted |

**Response `202`:**
```json
{
  "run_id": "CardCaptureFlow/1715523601234567",
  "video_id": "practice_session_03",
  "status": "pending",
  "created_at": "2026-05-12T14:00:00Z"
}
```

**Errors:** `404` (video not found), `409` (video already processing).

---

## 2. Runs

### `GET /api/v1/runs`

List all pipeline runs, newest first.

**Response `200`:**
```json
[
  {
    "run_id": "CardCaptureFlow/1715523601234567",
    "video_id": "practice_session_03",
    "status": "completed",
    "cards_extracted": 12,
    "elapsed_ms": 87430,
    "created_at": "2026-05-12T14:00:00Z"
  }
]
```

---

### `GET /api/v1/runs/{run_id}`

Get full run detail.

**Response `200`:**
```json
{
  "run_id": "CardCaptureFlow/1715523601234567",
  "video_id": "practice_session_03",
  "status": "completed",
  "config_json": {"detector": "docaligner", "config_preset": "balanced"},
  "cards_extracted": 12,
  "elapsed_ms": 87430,
  "created_at": "2026-05-12T14:00:00Z",
  "stage_timings": [
    {"stage_id": "detect", "elapsed_ms": 12000},
    {"stage_id": "novelty", "elapsed_ms": 300}
  ]
}
```

**Errors:** `404` if not found.

---

### `GET /api/v1/runs/{run_id}/cards`

Cards extracted in this run (lightweight list; use `/cards` with `run_id` filter for full pagination).

**Response `200`:** Array of Card summary objects (see §3).

---

### `GET /api/v1/runs/{run_id}/events`

Replay stored SSE events for a completed run as a JSON array.

**Response `200`:**
```json
[
  {"event": "stage_started", "data": {"stage_id": "detect", "run_id": "..."}, "ts": "2026-05-12T14:00:05Z"},
  {"event": "run_completed", "data": {"run_id": "...", "cards_extracted": 12}, "ts": "2026-05-12T14:01:27Z"}
]
```

---

### `GET /api/v1/runs/{run_id}/telemetry`

Per-stage timing and frame-count telemetry for this run.

**Response `200`:**
```json
{
  "run_id": "CardCaptureFlow/1715523601234567",
  "total_frames_sampled": 1420,
  "total_detections": 384,
  "stages": [
    {
      "stage_id": "detect",
      "frames_in": 1420,
      "frames_out": 384,
      "elapsed_ms": 12000,
      "throughput_fps": 32.1
    }
  ]
}
```

---

### `GET /api/v1/runs/{run_id}/rejection_log`

Every candidate dropped during this run, with stage and reason.

**Response `200`:**
```json
[
  {
    "frame_index": 4201,
    "stage_id": "novelty",
    "reason": "novelty_below_threshold",
    "thumbnail_url": "/static/runs/.../rejections/frame_4201_thumb.jpg"
  }
]
```

---

### `GET /api/v1/runs/{run_id}/hard_cases`

Hard cases auto-captured during this run.

**Response `200`:**
```json
[
  {
    "case_id": 7,
    "frame_index": 8830,
    "stage_id": "score",
    "reason": "low_confidence",
    "thumbnail_url": "/static/runs/.../hard_cases/case_7_thumb.jpg"
  }
]
```

---

## 3. Cards

### `GET /api/v1/cards`

Paginated, filtered card list.

**Query parameters:**

| Param | Type | Notes |
|---|---|---|
| `run_id` | string | Filter by pipeline run |
| `video_id` | string | Filter by source video |
| `dedup_group_id` | integer | Filter by dedup cluster |
| `review_state` | string | `"pending"` \| `"accepted"` \| `"rejected"` |
| `side` | string | `"front"` \| `"back"` |
| `is_foil` | boolean | |
| `confidence_min` | float | `[0.0, 1.0]` |
| `confidence_max` | float | `[0.0, 1.0]` |
| `page` | integer | Default `1` |
| `page_size` | integer | Default `50`, max `200` |

**Response `200`:**
```json
{
  "total": 124,
  "page": 1,
  "page_size": 50,
  "items": [
    {
      "card_id": "a1b2c3d4-...",
      "instance_id": "e5f6g7h8-...",
      "video_id": "practice_session_03",
      "run_id": "CardCaptureFlow/1715523601234567",
      "side": "front",
      "is_foil": false,
      "confidence": 0.91,
      "review_state": "pending",
      "canonical_url": "/static/crops/a1b2c3d4_canonical.jpg",
      "fused_url": "/static/crops/a1b2c3d4_fused.jpg",
      "created_at": "2026-05-12T14:01:10Z"
    }
  ]
}
```

---

### `GET /api/v1/cards/{card_id}`

Full card detail including source frames and detection metadata.

**Response `200`:**
```json
{
  "card_id": "a1b2c3d4-...",
  "instance_id": "e5f6g7h8-...",
  "video_id": "practice_session_03",
  "run_id": "CardCaptureFlow/1715523601234567",
  "side": "front",
  "is_foil": false,
  "confidence": 0.91,
  "review_state": "pending",
  "canonical_url": "/static/crops/a1b2c3d4_canonical.jpg",
  "fused_url": "/static/crops/a1b2c3d4_fused.jpg",
  "quality_score": {
    "sharpness": 0.82,
    "glare": 0.76,
    "aspect_ratio": 0.98,
    "size": 0.90,
    "complexity": 0.65,
    "border_purity": 0.88,
    "confidence": 0.91,
    "total": 0.84
  },
  "source_frame_indices": [4201, 4235, 4268],
  "dedup_group_id": 3,
  "created_at": "2026-05-12T14:01:10Z"
}
```

**Errors:** `404` if not found.

---

### `PATCH /api/v1/cards/{card_id}`

Update the review state of a card.

**Request body:**
```json
{
  "review_state": "accepted"
}
```

**Response `200`:** Updated Card detail object.

**Errors:** `404` if not found, `422` if `review_state` value is invalid.

---

### `POST /api/v1/cards/bulk`

Apply a verdict to multiple cards at once.

**Request body:**
```json
{
  "card_ids": ["a1b2c3d4-...", "b2c3d4e5-..."],
  "review_state": "rejected"
}
```

**Response `200`:**
```json
{
  "updated": 2,
  "failed": []
}
```

---

## 4. Label

### `GET /api/v1/label/truth/{video_id}`

Retrieve the current truth file for a video.

**Response `200`:** The `truth.json` payload (Contract 4 schema).

**Response `404`:** If no truth file exists yet.

---

### `PUT /api/v1/label/truth/{video_id}`

Create or overwrite the truth file for a video.

**Request body:** Full `truth.json` payload (Contract 4 schema).

**Response `200`:** The saved payload.

**Errors:** `422` if schema validation fails.

---

### `GET /api/v1/label/fb/next`

Return the next unlabeled card instance for the F/B trainer UX.

**Response `200`:**
```json
{
  "instance_id": "e5f6g7h8-...",
  "frame_index": 4201,
  "canonical_url": "/static/crops/e5f6g7h8_canonical.jpg",
  "video_id": "practice_session_03",
  "run_id": "CardCaptureFlow/1715523601234567",
  "labels_collected": 247,
  "labels_target": 500
}
```

**Response `204`:** No unlabeled instances remain.

---

### `POST /api/v1/label/fb`

Submit a single F/B label (one keypress in the trainer UX).

**Request body:**
```json
{
  "instance_id": "e5f6g7h8-...",
  "frame_index": 4201,
  "side": "front"
}
```

**Response `201`:**
```json
{
  "label_id": 248,
  "instance_id": "e5f6g7h8-...",
  "side": "front",
  "created_at": "2026-05-12T15:00:01Z"
}
```

**Errors:** `422` if `side` not in `["front", "back", "uncertain"]`.

---

### `GET /api/v1/label/clusters`

List dedup clusters for operator review.

**Query parameters:** `status` (`"unverified"` \| `"confirmed"` \| `"split"` \| `"merged"`), `page`, `page_size`.

**Response `200`:**
```json
{
  "total": 23,
  "page": 1,
  "page_size": 20,
  "items": [
    {
      "cluster_id": 5,
      "status": "unverified",
      "predicted_member_ids": ["e5f6g7h8-...", "a1b2c3d4-...", "b2c3d4e5-..."],
      "confirmed_member_ids": null,
      "member_thumbnails": ["/static/crops/e5f6_thumb.jpg", "..."],
      "updated_at": "2026-05-12T14:01:10Z"
    }
  ]
}
```

---

### `PATCH /api/v1/label/clusters/{cluster_id}`

Update a dedup cluster (confirm, split, or merge).

**Request body:**
```json
{
  "status": "confirmed",
  "confirmed_member_ids": ["e5f6g7h8-...", "a1b2c3d4-..."]
}
```

**Response `200`:** Updated cluster object (same shape as list item above).

**Errors:** `404` if not found, `422` on invalid status transition.

---

## 5. Training

### `GET /api/v1/training/datasets`

List available training datasets and their statistics.

**Response `200`:**
```json
[
  {
    "model_name": "fb_classifier",
    "total_labels": 247,
    "class_distribution": {"front": 130, "back": 112, "uncertain": 5},
    "last_updated": "2026-05-12T15:00:01Z"
  }
]
```

---

### `POST /api/v1/training/retrain/{model_name}`

Trigger a local retrain for a model.

**Request body:**
```json
{
  "epochs": 20,
  "learning_rate": 0.001
}
```

**Response `202`:**
```json
{
  "job_id": "retrain-fb_classifier-1715527201",
  "model_name": "fb_classifier",
  "status": "queued",
  "created_at": "2026-05-12T15:00:01Z"
}
```

**Errors:** `404` if `model_name` unknown, `409` if a retrain is already running.

---

### `GET /api/v1/training/jobs/{job_id}`

Get status of a training job.

**Response `200`:**
```json
{
  "job_id": "retrain-fb_classifier-1715527201",
  "model_name": "fb_classifier",
  "status": "running",
  "progress": {"epoch": 8, "total_epochs": 20, "val_accuracy": 0.89},
  "created_at": "2026-05-12T15:00:01Z",
  "completed_at": null
}
```

**`status` values:** `"queued"` \| `"running"` \| `"completed"` \| `"failed"`

---

## 6. Regression

### `GET /api/v1/regression/baselines`

List all named regression baselines.

**Response `200`:**
```json
[
  {
    "baseline_id": 1,
    "name": "baseline_v4.1",
    "code_sha": "e34f40d",
    "created_at": "2026-05-12T14:00:00Z"
  }
]
```

---

### `POST /api/v1/regression/baselines`

Promote the current pipeline state as a new named baseline.

**Request body:**
```json
{
  "name": "baseline_v4.2",
  "code_sha": "abc1234"
}
```

**Response `201`:** Single baseline object (same shape as list item above).

**Errors:** `409` if `name` already exists.

---

### `POST /api/v1/regression/run`

Start a regression harness run against a baseline.

**Request body:**
```json
{
  "baseline_id": 1,
  "video_subset": ["practice_session_03", "practice_session_05"]
}
```

**Response `202`:**
```json
{
  "run_id": 7,
  "baseline_id": 1,
  "status": "running",
  "created_at": "2026-05-12T16:00:00Z"
}
```

---

### `GET /api/v1/regression/runs/{run_id}`

Get results of a regression run.

**Response `200`:**
```json
{
  "run_id": 7,
  "baseline_id": 1,
  "status": "completed",
  "metrics": {
    "card_recall": 0.92,
    "card_precision": 0.88,
    "side_accuracy": 0.79,
    "dedup_ari": 0.81,
    "image_quality_ssim": 0.87
  },
  "per_video": [
    {
      "video_id": "practice_session_03",
      "card_recall": 0.91,
      "card_precision": 0.90,
      "side_accuracy": 0.80,
      "regressions": []
    }
  ],
  "created_at": "2026-05-12T16:00:00Z"
}
```

---

### `GET /api/v1/regression/compare`

Compare two regression runs side by side.

**Query parameters:** `a` (run_id), `b` (run_id)

**Response `200`:**
```json
{
  "run_a": 6,
  "run_b": 7,
  "metric_deltas": {
    "card_recall": +0.02,
    "card_precision": -0.01,
    "side_accuracy": +0.05,
    "dedup_ari": +0.03,
    "image_quality_ssim": 0.00
  },
  "regressions": [],
  "per_video_deltas": [
    {
      "video_id": "practice_session_03",
      "card_recall_delta": +0.02,
      "is_regression": false
    }
  ]
}
```

---

## 7. Config

### `GET /api/v1/config/presets`

List all config presets.

**Response `200`:**
```json
[
  {
    "preset_name": "balanced",
    "description": "Default balanced trade-off between speed and quality",
    "config": {
      "corner_confidence": 0.5,
      "background_novelty_threshold": 0.08,
      "centroid_jump_ratio": 0.30,
      "valley_drop_ratio": 0.40,
      "foil_threshold": 50.0
    }
  }
]
```

---

### `POST /api/v1/config/presets`

Create a custom config preset.

**Request body:**
```json
{
  "preset_name": "high_recall",
  "description": "Lower confidence thresholds for difficult lighting",
  "config": {
    "corner_confidence": 0.35,
    "background_novelty_threshold": 0.06
  }
}
```

**Response `201`:** Created preset object (same shape as list item above).

**Errors:** `409` if `preset_name` already exists.

---

### `GET /api/v1/config/playground/{run_id}`

Retrieve persisted Metaflow artifact data for the threshold playground UI.

**Response `200`:**
```json
{
  "run_id": "CardCaptureFlow/1715523601234567",
  "available_steps": ["novelty", "score", "dedup"],
  "slider_params": [
    {
      "param": "background_novelty_threshold",
      "current": 0.08,
      "min": 0.02,
      "max": 0.30,
      "step": 0.01,
      "affects_steps": ["novelty", "score"]
    },
    {
      "param": "corner_confidence",
      "current": 0.50,
      "min": 0.20,
      "max": 0.90,
      "step": 0.05,
      "affects_steps": ["novelty"]
    }
  ]
}
```

---

## 8. SSE — Real-Time Progress

**Endpoint:** `GET /events/{run_id}`

**Content-Type:** `text/event-stream`

The server emits newline-delimited SSE events as each Metaflow step completes. The `event:` field identifies the event type; `data:` is a JSON object.

### Event: `stage_started`

```
event: stage_started
data: {"run_id": "CardCaptureFlow/1715523601234567", "stage_id": "detect", "ts": "2026-05-12T14:00:05Z"}
```

### Event: `stage_progress`

```
event: stage_progress
data: {"run_id": "CardCaptureFlow/1715523601234567", "stage_id": "detect", "pct": 42, "detail": "Sampling frame 600/1420", "ts": "2026-05-12T14:00:11Z"}
```

**Fields:** `pct` is `0–100` (integer); `detail` is a human-readable status string.

### Event: `stage_completed`

```
event: stage_completed
data: {"run_id": "CardCaptureFlow/1715523601234567", "stage_id": "detect", "elapsed_ms": 12100, "ts": "2026-05-12T14:00:17Z"}
```

### Event: `artifact_persisted`

```
event: artifact_persisted
data: {"run_id": "CardCaptureFlow/1715523601234567", "stage_id": "detect", "artifact_name": "corner_detections", "artifact_ref": "CardCaptureFlow/1715523601234567/detect/corner_detections", "ts": "2026-05-12T14:00:17Z"}
```

### Event: `run_completed`

```
event: run_completed
data: {"run_id": "CardCaptureFlow/1715523601234567", "cards_extracted": 12, "elapsed_ms": 87430, "ts": "2026-05-12T14:01:27Z"}
```

### Event: `run_failed`

```
event: run_failed
data: {"run_id": "CardCaptureFlow/1715523601234567", "stage_id": "refine", "error": "GPU out of memory", "ts": "2026-05-12T14:00:55Z"}
```

### SSE Ordering Guarantee

Events are emitted in pipeline step order. `stage_started` always precedes `artifact_persisted` and `stage_completed` for the same step. `run_completed` or `run_failed` is always the last event.

---

## Change Policy

Changes to this contract require explicit written ack from all four surface owners (A, B, C, D) before any code is merged.
