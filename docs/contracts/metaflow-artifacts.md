# Contract 3 — Metaflow Artifact Contract

**Status:** Frozen (Wave 1 sign-off)  
**Owner:** Surface A (Orchestration / Pipeline)  
**Consumers:** Surface B (threshold playground reads persisted artifacts), Surface D (harness reads `final_cards`, `dedup_groups`)

> **Stability guarantee.** The step names, artifact names, and artifact types listed here will not change without explicit four-surface ack. New artifacts may be added to existing steps; they require four-surface ack before landing. Steps may not be renamed, split, or merged without four-surface ack.

---

## Overview

`pipeline/card_capture_flow.py` is a Metaflow `FlowSpec` containing one `@step` per logical pipeline stage (plus `start`, `fuse_fanout`, `fuse_join`, and `end`). The flow stays ≤ 200 lines; all stage logic lives in `pipeline/steps/<name>.py`.

Stages 1–3 (sampler → triage → YOLO detector) run inside the `detect` step using the existing `multiprocessing` + bounded `Queue` streaming subsystem. Metaflow sees this block as a single opaque step. Stage 9 (per-track fusion) is a Metaflow `foreach` branch (`fuse_fanout` → `fuse` [×N] → `fuse_join`).

---

## Step Graph

```
start
  └─ detect
       └─ novelty
            └─ track
                 └─ refine
                      └─ score
                           └─ resolve
                                └─ fuse_fanout
                                     └─ fuse [foreach, one per prepared_track]
                                          └─ fuse_join
                                               └─ dedup
                                                    └─ store
                                                         └─ end
```

---

## Named Artifacts Per Step

Every artifact listed here is set as `self.<artifact_name> = <value>` inside its step and is automatically persisted by Metaflow to the local datastore (default: `~/.metaflow/`). Downstream steps and external tools (threshold playground, harness) read artifacts by step name and artifact name using the Metaflow client API.

### `start`

| Artifact | Python type | Description |
|---|---|---|
| `run_context` | `pipeline.contracts.RunContext` | Paths, config, telemetry sink, Metaflow run-id |

### `detect`

Wraps Stages 1 (sampler), 2 (frame triage), and 3 (YOLO corner detection).

| Artifact | Python type | Description |
|---|---|---|
| `frame_samples` | `list[pipeline.contracts.FrameSample]` | High-res frame indices + timestamps selected by the sampler |
| `triaged_frames` | `list[pipeline.contracts.TriagedFrame]` | Frames passing the empty-frame triage gate (Stage 2) |
| `corner_detections` | `list[pipeline.contracts.CornerDetection]` | YOLO-OBB 4-corner detections with confidence |

**Note:** `np.ndarray` images are not persisted directly. `FrameSample.image_path` and `CornerDetection.frame_index` reference on-disk source frames in `output_dir/frames/`.

### `novelty`

Stage 4 — background novelty gate.

| Artifact | Python type | Description |
|---|---|---|
| `novelty_filtered_candidates` | `list[pipeline.contracts.NoveltyFilteredCandidate]` | Candidates that pass the quad-interior novelty gate |
| `background_model` | `pipeline.contracts.BackgroundModel` | Serialized mean grayscale background reference (path to `.npy` file) |

### `track`

Stage 5 — session-aware tracking (BoT-SORT or ByteTrack backend).

| Artifact | Python type | Description |
|---|---|---|
| `tracks` | `list[pipeline.contracts.Track]` | Active tracks, each with `instance_id` (UUID-4), candidate list, and last centroid |
| `session_resets` | `list[pipeline.contracts.SessionReset]` | Reset events with `reason` (`"frame_gap"` \| `"valley_split"` \| `"centroid_jump"` \| `"reid_shift"`) and `frame_index` |

### `refine`

Stage 6 — GPU/CPU perspective warp to 750×1050.

| Artifact | Python type | Description |
|---|---|---|
| `rectified_crops` | `list[pipeline.contracts.RectifiedCrop]` | Perspective-corrected crops at 750×1050; stored as paths under `output_dir/crops/` |

### `score`

Stage 7 — quality scoring and track pruning.

| Artifact | Python type | Description |
|---|---|---|
| `scored_candidates` | `list[pipeline.contracts.ScoredCandidate]` | Candidates with attached `QualityScore` (7 components + total) |
| `pruned_tracks` | `list[str]` | `instance_id` strings of tracks dropped by median quad-novelty pruning |

### `resolve`

Stage 8 — Front/Back resolution and pHash gating.

| Artifact | Python type | Description |
|---|---|---|
| `prepared_tracks` | `list[pipeline.contracts.PreparedTrack]` | Tracks annotated with `angle` (`"front"` \| `"back"`), `session_id`, `primary_hash` (pHash hex string), and `side_score` |

### `fuse` (foreach branch)

Stage 9 — per-track lighting-diverse selection + median/glare-rejection fusion. One `fuse` step execution per element of `prepared_tracks` (Metaflow `foreach`).

| Artifact | Python type | Description |
|---|---|---|
| `fused_canonical` | `pipeline.contracts.FusedCanonical` | Single fused 750×1050 image path + metadata for one track |

After `fuse_join`, the merged list is:

| Artifact | Python type | Description |
|---|---|---|
| `fused_canonicals` | `list[pipeline.contracts.FusedCanonical]` | All per-track fused images (joined from the foreach branches) |

### `dedup`

Stage 10a — pHash + ReID deduplication.

| Artifact | Python type | Description |
|---|---|---|
| `dedup_groups` | `list[pipeline.contracts.DedupGroup]` | Clusters of `instance_id` strings predicted to be the same physical card |
| `dedup_distances` | `str` | Path to a `.npz` sparse matrix of pairwise cosine / Hamming distances |

### `store`

Stage 10b — SQLite persistence.

| Artifact | Python type | Description |
|---|---|---|
| `final_cards` | `list[pipeline.contracts.FinalCard]` | Records written to `cards.sqlite`; each includes `card_id`, `instance_id`, `side`, `canonical_path`, `fused_path`, `dedup_group_id` |

---

## Contract Types Reference

All types are defined in `pipeline/contracts.py` as frozen Pydantic v2 models. The canonical definitions live in that file; this section is a summary for cross-surface readability.

```python
class RunContext(BaseModel):
    video_path: str
    output_dir: str
    db_path: str
    detector: str
    config_preset: str
    metaflow_run_id: str

class FrameSample(BaseModel):
    frame_index: int
    timestamp_ms: int
    image_path: str          # relative to output_dir/frames/
    w: int
    h: int

class TriagedFrame(BaseModel):
    frame_index: int
    timestamp_ms: int
    image_path: str
    triage_metrics: dict     # sharpness, empty_ratio, etc.

class CornerDetection(BaseModel):
    frame_index: int
    timestamp_ms: int
    corners: list[list[float]]  # [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    confidence: float
    w: int
    h: int

class BackgroundModel(BaseModel):
    npy_path: str            # path to saved numpy array

class NoveltyFilteredCandidate(BaseModel):
    detection_id: str
    frame_index: int
    timestamp_ms: int
    corners: list[list[float]]
    confidence: float
    novelty_score: float

class Track(BaseModel):
    instance_id: str         # UUID-4
    candidate_ids: list[str] # detection_id strings
    last_centroid: list[float]
    last_frame_index: int
    active: bool
    angle: str               # "front" | "back" | "unknown"

class SessionReset(BaseModel):
    frame_index: int
    reason: str              # "frame_gap" | "valley_split" | "centroid_jump" | "reid_shift"
    session_id: int

class RectifiedCrop(BaseModel):
    instance_id: str
    frame_index: int
    crop_path: str           # relative to output_dir/crops/
    w: int                   # 750
    h: int                   # 1050

class QualityScore(BaseModel):
    sharpness: float
    glare: float
    aspect_ratio: float
    size: float
    complexity: float
    border_purity: float
    confidence: float
    total: float

class ScoredCandidate(BaseModel):
    detection_id: str
    instance_id: str
    frame_index: int
    crop_path: str
    score: QualityScore

class PreparedTrack(BaseModel):
    instance_id: str
    session_id: int
    angle: str               # "front" | "back"
    primary_hash: str        # pHash hex string
    side_score: float
    canonical_crop_paths: list[str]

class FusedCanonical(BaseModel):
    instance_id: str
    fused_path: str          # path to 750×1050 fused image
    fusion_method: str       # "median" | "glare_rejection"
    is_foil: bool
    selected_frame_indices: list[int]

class DedupGroup(BaseModel):
    group_id: str            # UUID-4 or integer string
    instance_ids: list[str]
    representative_id: str   # the instance_id chosen as canonical for the group

class FinalCard(BaseModel):
    card_id: str
    instance_id: str
    side: str                # "front" | "back"
    is_foil: bool
    canonical_path: str
    fused_path: str
    dedup_group_id: str | None
    quality_total: float
```

---

## Reading Artifacts from External Code

Surface B (threshold playground) and Surface D (harness) access artifacts using the Metaflow client:

```python
from metaflow import Run

run = Run("CardCaptureFlow/1715523601234567")

# Access artifacts by step name and artifact name:
corner_detections = run["detect"].task.data.corner_detections
scored_candidates  = run["score"].task.data.scored_candidates
final_cards        = run["store"].task.data.final_cards

# For foreach steps, iterate branches:
fused_canonicals = [step.task.data.fused_canonical for step in run["fuse"].tasks()]
```

The threshold playground uses this pattern to recompute only the steps downstream of the slider being dragged, without re-running the detector or sampler.

---

## Stability Rules

1. **Step names are frozen.** Renaming, splitting, or merging steps breaks the Metaflow client access pattern used by B and D.
2. **Artifact names are frozen.** Consumers reference artifacts by string name; renames are breaking changes.
3. **Artifact types may gain optional fields.** Adding a new optional field to a Pydantic model is non-breaking. Removing or renaming a field is breaking.
4. **The `foreach` branch for `fuse` is the only parallel step.** Adding new `foreach` branches requires four-surface ack.

---

## Change Policy

Changes to this contract require explicit written ack from all four surface owners (A, B, C, D) before any code is merged.
