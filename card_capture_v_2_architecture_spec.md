# Card Capture v2 Architecture & Implementation Specification

**Project:** Sports Trading Card Image Extraction from Video  
**Status:** Draft implementation specification  
**Primary Goal:** Extend the current MVP into a future-proof pipeline that supports canonical card images, multi-frame enhancement, multi-angle evidence capture, and future defect/grading models without requiring large refactors later.

---

## 1. Purpose

Card Capture currently extracts high-quality stills of trading cards from handheld lightbox videos. The current MVP performs frame sampling, card detection, cropping, quality scoring, selection, storage, and web-based review.

This v2 specification expands the architecture so the system can:

1. Detect and group frames belonging to the same physical card instance.
2. Produce consistent vertical, perspective-corrected card images.
3. Preserve raw evidence frames for future grading and defect analysis.
4. Generate optional multi-frame composites for clean review/display images.
5. Collect multiple lighting and viewing angles when present in the source video.
6. Store diagnostic and metadata-rich artifacts for future scoring, model training, and quality analysis.
7. Avoid implementing grading/scoring models now, while collecting the data needed to build them later.

The intent is to implement architectural support now so that future grading, defect detection, model training, and image enhancement features can be added incrementally.

---

## 2. Design Principles

### 2.1 Preserve Evidence

Do not destroy or overwrite original visual evidence in the pursuit of cleaner-looking images.

The system should preserve:

- raw source frames used for each card instance
- lightly processed rectified views
- normalized review images
- optional fused display images
- metadata describing pose, lighting, glare, sharpness, exposure, and transformations

Future grading workflows must be able to inspect original or minimally processed evidence.

### 2.2 Separate Review Output From Evidence Output

The system should produce different classes of outputs for different purposes:

| Output Type | Purpose | Processing Level |
|---|---|---|
| Raw frame | Audit trail and future model training | None |
| Rectified raw card | Standard geometry while preserving appearance | Perspective correction only |
| Normalized review image | Human-friendly browsing/review | Color/exposure normalization allowed |
| Fused display image | Best visual representation | Multi-frame enhancement allowed |
| Evidence view set | Future grading and defect detection | Minimal or controlled processing |

### 2.3 Collect Now, Score Later

This version should not implement grading scores or defect scores.

Instead, it should collect and store the measurements needed to train or implement scoring later:

- sharpness
- exposure
- glare ratio
- glare centroid
- pose / homography
- frame timestamp
- crop geometry
- lighting cluster
- pose cluster
- raw/rectified image paths
- detection confidence
- rejection reasons, where relevant

### 2.4 Build Around Card Instances

The central domain object should become a **CardInstance**, not an individual frame or detection.

A CardInstance represents one observed physical card side over a contiguous time segment in a video.

Each CardInstance may contain many CardViews.

### 2.5 Prefer Sequential Video Processing

Avoid expensive random-access seeking wherever possible. Decode video sequentially and collect metrics, frames, and candidate windows in one pass when feasible.

### 2.6 Make Everything Configurable

Thresholds, model paths, batch sizes, output modes, diagnostic settings, and feature toggles should live in configuration rather than hardcoded constants.

---

## 3. High-Level v2 Pipeline

```text
Input Video
  ↓
[Video Decode + Metric Scan]
  ↓
Frame Metrics Timeline
  ↓
[Card Presence Window Detection]
  ↓
Candidate Card Windows
  ↓
[Candidate Frame Selection]
  ↓
Candidate Frames per Window
  ↓
[Batch Card Detection]
  ↓
Frame-Level Card Detections
  ↓
[Corner Refinement + Geometry Validation]
  ↓
Rectifiable Card Views
  ↓
[Card Instance Grouping]
  ↓
CardInstance objects with many CardView objects
  ↓
[Perspective Rectification + Orientation Normalization]
  ↓
Canonical Card Views
  ↓
[View Characterization]
  ↓
Pose, lighting, glare, sharpness, exposure metadata
  ↓
[Representative View Selection]
  ↓
Evidence View Set per CardInstance
  ↓
[Optional Multi-Frame Fusion]
  ↓
Fused Display Image
  ↓
[Storage + Diagnostics + Review UI]
```

---

## 4. Core Concepts

## 4.1 Video

A Video is the top-level input asset.

It contains:

- source file path
- duration
- frame rate
- resolution
- codec metadata
- processing configuration
- processing status
- derived card instances

## 4.2 FrameMetric

A FrameMetric is a lightweight record describing measurements from a sampled or decoded frame.

It should be cheap to compute and useful for diagnostics, sampling, and later model training.

Suggested fields:

```python
@dataclass
class FrameMetric:
    video_id: str
    frame_index: int
    timestamp_ms: int

    width: int
    height: int

    variance: float | None = None
    sharpness: float | None = None
    motion: float | None = None
    edge_density: float | None = None
    exposure_mean: float | None = None
    exposure_p95: float | None = None
    exposure_p99: float | None = None
    glare_ratio: float | None = None

    presence_score: float | None = None
    selected_for_detection: bool = False
    window_id: str | None = None

    metadata: dict = field(default_factory=dict)
```

## 4.3 CardPresenceWindow

A CardPresenceWindow represents a contiguous time region where the system believes a card is present.

Suggested fields:

```python
@dataclass
class CardPresenceWindow:
    window_id: str
    video_id: str
    start_frame_index: int
    end_frame_index: int
    start_timestamp_ms: int
    end_timestamp_ms: int

    confidence: float
    source: str  # e.g. "adaptive_contrast", "detector_guided", "manual"

    frame_metric_count: int
    selected_frame_indices: list[int]
    metadata: dict = field(default_factory=dict)
```

## 4.4 FrameSample

FrameSample may continue to exist, but should be treated as an image-bearing object selected from a window.

Suggested updates:

```python
@dataclass
class FrameSample:
    video_id: str
    frame_index: int
    timestamp_ms: int
    image: np.ndarray
    width: int
    height: int

    window_id: str | None = None
    metrics: FrameMetric | None = None
    source: str = "sampler"
    metadata: dict = field(default_factory=dict)
```

## 4.5 CardDetection

A CardDetection is a frame-level model or CV result indicating where a card appears in a specific frame.

Suggested updates:

```python
@dataclass
class CardDetection:
    detection_id: str
    video_id: str
    frame_index: int
    timestamp_ms: int
    window_id: str | None

    polygon: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    confidence: float
    label: str
    detector_name: str
    detector_version: str | None = None

    geometry_valid: bool | None = None
    rejection_reason: str | None = None
    metadata: dict = field(default_factory=dict)
```

## 4.6 CardView

A CardView is one usable view of a card from one frame.

This is the key object for future grading. It should include both file paths and metadata.

Suggested fields:

```python
@dataclass
class CardView:
    view_id: str
    instance_id: str | None
    video_id: str
    frame_index: int
    timestamp_ms: int

    detection_id: str
    window_id: str | None

    raw_frame_path: str | None
    raw_crop_path: str | None
    rectified_raw_path: str | None
    normalized_path: str | None

    polygon: list[tuple[float, float]]
    homography: list[list[float]] | None
    canonical_width: int
    canonical_height: int
    orientation_degrees: int | None

    sharpness: float | None = None
    exposure_mean: float | None = None
    exposure_p95: float | None = None
    glare_ratio: float | None = None
    glare_centroid_x: float | None = None
    glare_centroid_y: float | None = None

    pose_features: dict = field(default_factory=dict)
    lighting_features: dict = field(default_factory=dict)

    pose_cluster: str | None = None
    lighting_cluster: str | None = None
    representative_role: str | None = None
    # examples: "canonical", "low_glare", "specular", "left_tilt", "right_tilt", "raw_evidence"

    metadata: dict = field(default_factory=dict)
```

## 4.7 CardInstance

A CardInstance represents a single observed card side in the video.

It groups many CardViews.

Suggested fields:

```python
@dataclass
class CardInstance:
    instance_id: str
    video_id: str

    start_timestamp_ms: int
    end_timestamp_ms: int
    start_frame_index: int
    end_frame_index: int

    side: str | None = None  # "front", "back", or None until classified/reviewed
    status: str = "pending_review"

    canonical_view_id: str | None = None
    canonical_image_path: str | None = None
    fused_display_image_path: str | None = None

    evidence_view_ids: list[str] = field(default_factory=list)
    all_view_ids: list[str] = field(default_factory=list)

    detection_count: int = 0
    view_count: int = 0

    metadata: dict = field(default_factory=dict)
```

## 4.8 MultiFrameComposite

A MultiFrameComposite is an optional derived image built from multiple aligned CardViews.

It should be treated as a display/review asset, not the only source of grading evidence.

```python
@dataclass
class MultiFrameComposite:
    composite_id: str
    instance_id: str
    source_view_ids: list[str]

    output_path: str
    method: str  # "weighted_average", "median", "trimmed_mean", "super_resolution"

    alignment_success: bool
    alignment_error: float | None = None

    metadata: dict = field(default_factory=dict)
```

---

## 5. Pipeline Stages

## 5.1 Stage 1: Sequential Video Decode + Metric Scan

### Purpose

Decode the video in order, compute lightweight metrics, and avoid expensive random-access seeking.

### Requirements

- Read frames sequentially whenever possible.
- Support configurable scan FPS.
- Support low-resolution metric computation.
- Store a metric timeline for diagnostics and future analysis.
- Avoid loading the entire video into memory.

### Metrics to Collect

At minimum:

- RGB/color variance
- sharpness estimate
- exposure mean
- exposure p95 / p99
- glare ratio
- optional edge density
- optional motion estimate

### Notes

Motion should not be used as a positive card-presence signal by default. Camera movement can create false positives. Motion is more useful as a stability penalty when selecting frames.

### Output

- `FrameMetric` records
- optional metric timeline CSV
- optional metric visualization

---

## 5.2 Stage 2: Adaptive Card Presence Window Detection

### Purpose

Identify contiguous video segments where a card is likely present.

### Recommended Strategy

Replace fixed contrast thresholds with adaptive thresholds.

Example:

```python
presence_threshold = median(variance) + k * median_absolute_deviation(variance)
```

Alternative:

```python
presence_threshold = percentile(variance, configured_percentile)
```

### Presence Score

Suggested initial formula:

```text
presence_score =
    w_variance * normalized_variance
  + w_edge * normalized_edge_density
  - w_motion * normalized_motion
```

For the initial implementation:

- variance should be the primary positive signal
- edge density may be collected but should not be relied on until calibrated
- motion should be a penalty or diagnostic, not an OR-trigger

### Window Rules

Configurable parameters:

```yaml
sampler:
  min_window_ms: 700
  max_window_ms: 5000
  merge_gap_ms: 300
  min_presence_frames: 3
```

### Output

- `CardPresenceWindow` records
- diagnostic reasons for accepted/rejected windows

---

## 5.3 Stage 3: Candidate Frame Selection

### Purpose

Select a small number of promising frames from each card presence window.

### Requirements

For each window, select top K frames based on:

- sharpness
- low motion
- acceptable exposure
- low severe glare for canonical output
- optional diversity of pose/lighting for evidence output

### Candidate Types

Candidate frames should be selected for multiple purposes:

| Candidate Role | Selection Bias |
|---|---|
| canonical_candidate | sharp, low glare, centered, stable |
| evidence_candidate | diverse pose and lighting |
| specular_candidate | visible glare/specular changes, if useful |
| fusion_candidate | nearby, alignable, sharp frames |

### Output

- `FrameSample` objects
- candidate role metadata

---

## 5.4 Stage 4: Batch Card Detection

### Purpose

Run detector inference efficiently across candidate frames.

### Requirements

- Add `detect_batch` to detector interface.
- Use adaptive batch sizing.
- Preserve compatibility with single-frame `detect`.
- Record detector name, version, confidence threshold, and model path.

### Interface

```python
class CardDetector:
    def detect(self, frame: FrameSample) -> list[CardDetection]:
        ...

    def detect_batch(self, frames: list[FrameSample]) -> list[list[CardDetection]]:
        ...
```

### Adaptive Batch Sizing

Pseudo-logic:

```python
batch_size = configured_batch_size
while True:
    try:
        run_batch(batch_size)
        maybe_increase_if_auto()
    except OutOfMemoryError:
        batch_size = max(1, batch_size // 2)
```

### Output

- `CardDetection` records
- detector diagnostics

---

## 5.5 Stage 5: Corner Refinement + Geometry Validation

### Purpose

Convert rough detections into reliable card quadrilaterals suitable for perspective rectification.

### Requirements

A YOLO bounding box alone is not enough for canonical card exports. The system needs a robust card polygon.

### Recommended Approach

Initial implementation:

1. Use detector bounding box as region of interest.
2. Run edge detection / contour detection inside ROI.
3. Find quadrilateral candidates.
4. Validate aspect ratio and rectangularity.
5. Fall back to detector box if corner refinement fails, but mark confidence lower.

Future implementation:

- train segmentation model
- train corner-regression model
- use keypoint detector for four card corners

### Geometry Validation Checks

- aspect ratio near expected card ratio
- polygon roughly convex
- area within expected range
- card not too close to frame boundary
- corner confidence sufficient
- perspective distortion not extreme unless intentionally captured

### Output

- refined polygon
- homography inputs
- validation status
- rejection reason if invalid

---

## 5.6 Stage 6: Perspective Rectification + Orientation Normalization

### Purpose

Produce consistent vertical card images.

### Requirements

For each valid CardDetection:

1. Estimate four card corners.
2. Order corners consistently.
3. Compute homography.
4. Warp card into fixed portrait canvas.
5. Rotate to vertical orientation.
6. Save rectified raw card image.
7. Optionally save normalized review image.

### Canonical Dimensions

Use configurable dimensions.

Example:

```yaml
canonical_image:
  width: 1125
  height: 1575
  aspect_ratio: 0.7142857
```

### Orientation Detection

Initial options:

- use longest side to enforce portrait layout
- optionally use OCR/text orientation later
- optionally use ML side/orientation classifier later

### Output Files

For each CardView:

```text
views/{instance_id}/{view_id}_rectified_raw.jpg
views/{instance_id}/{view_id}_normalized.jpg
```

---

## 5.7 Stage 7: Card Instance Grouping

### Purpose

Group multiple detections/views from the same physical card side into a single CardInstance.

### Grouping Signals

Use a combination of:

- temporal proximity
- same presence window
- perceptual image similarity
- rectified crop similarity
- detection geometry consistency
- optional OCR/card identity later

### Initial Grouping Strategy

1. Group by presence window.
2. Within each window, cluster detections by temporal continuity and image similarity.
3. Create one CardInstance per cluster.
4. Avoid merging front and back if appearance differs significantly.

### Perceptual Hashing

Compute pHash or dHash for rectified crops.

Use hash distance to reduce duplicates, but do not merge too aggressively.

Suggested rule:

```text
same_instance =
    same_window_or_nearby_time
    AND hash_distance < threshold
```

Do not rely only on hash distance because card backs may be similar across different cards.

### Output

- `CardInstance` records
- assigned `CardView.instance_id`

---

## 5.8 Stage 8: View Characterization

### Purpose

Collect metadata about each CardView for future scoring, grading, clustering, and model training.

### Required Measurements

Collect but do not convert into grading scores yet.

#### Sharpness

- Laplacian variance or equivalent
- optional edge coherence

#### Exposure

- mean brightness
- p95 brightness
- p99 brightness
- underexposed pixel ratio
- overexposed pixel ratio

#### Glare

- glare area ratio
- glare centroid x/y in canonical coordinates
- glare bounding box
- glare intensity statistics

#### Pose

From homography or polygon:

- perspective skew
- apparent tilt x/y proxy
- card area ratio in source frame
- corner positions
- rotation angle

#### Lighting

- brightness gradient direction
- highlight location
- specular area movement across frames
- color temperature / white balance estimate

### Output

- populated fields on CardView
- optional `view_metrics` table
- diagnostic JSON

---

## 5.9 Stage 9: Representative View Selection

### Purpose

Select a compact but diverse evidence set per CardInstance.

This is not grading. It is evidence collection.

### Recommended Representative Roles

Each CardInstance should try to select:

| Role | Description |
|---|---|
| canonical | sharp, low-glare, stable, best general-purpose view |
| low_glare | lowest glare among sharp candidates |
| specular | frame with useful reflective highlight, if present |
| left_tilt | representative tilted/angled view, if present |
| right_tilt | representative opposite tilted/angled view, if present |
| high_detail | locally sharpest/detail-rich view |
| raw_evidence | minimally processed source-preserving view |

Not every card will have every role.

### Clustering Strategy

Cluster CardViews by:

- pose features
- lighting features
- glare centroid
- sharpness
- timestamp

Then pick high-quality representatives from distinct clusters.

### Output

- `representative_role` field on selected CardViews
- `CardInstance.evidence_view_ids`
- review UI can show representative views grouped by role

---

## 5.10 Stage 10: Optional Multi-Frame Fusion

### Purpose

Create a clean display image using several aligned frames.

This image is for human review and presentation. It should not replace evidence views.

### Requirements

For each CardInstance:

1. Select fusion candidate CardViews.
2. Ensure they are alignable in canonical space.
3. Align using homography or image registration.
4. Fuse with a robust method.
5. Save fused display image.
6. Record source views and method.

### Initial Fusion Methods

Start simple:

- weighted average
- median
- trimmed mean

Possible future methods:

- burst super-resolution
- neural deblurring
- local best-region compositing

### Fusion Candidate Selection

Prefer frames that are:

- close in time
- similar pose
- sharp
- low motion blur
- not severely overexposed
- successfully rectified

### Important Constraint

Never use only the fused image for future defect detection. Fusion may hide subtle scratches, dents, print lines, or surface defects.

---

## 5.11 Stage 11: Lighting Normalization

### Purpose

Produce visually consistent review images while preserving raw evidence separately.

### Safe Normalization Options

For normalized review images:

- white balance correction
- exposure normalization
- mild contrast normalization
- background masking
- gentle tone mapping

### Avoid or Gate Behind Config

- aggressive glare removal
- heavy denoising
- hallucination-based enhancement
- generative restoration
- aggressive sharpening

### Required Output Separation

Always keep:

- rectified raw image
- normalized review image, if generated

Do not overwrite raw or rectified raw artifacts.

---

## 6. Storage Design

## 6.1 File Layout

Suggested output directory structure:

```text
output/
  videos/
    {video_id}/
      source_metadata.json
      processing_config.yaml
      diagnostics/
        frame_metrics.csv
        windows.json
        detections.json
        card_instances.json
        timeline.png
        rejection_summary.json
      frames/
        sampled/
        rejected/
      instances/
        {instance_id}/
          instance.json
          canonical.jpg
          fused_display.jpg
          views/
            {view_id}_raw_frame.jpg
            {view_id}_raw_crop.jpg
            {view_id}_rectified_raw.jpg
            {view_id}_normalized.jpg
          evidence/
            canonical.jpg
            low_glare.jpg
            specular.jpg
            left_tilt.jpg
            right_tilt.jpg
```

## 6.2 Database Tables

The existing SQLite schema should be extended rather than replaced.

### videos

Add or ensure:

- video_id
- source_path
- duration_ms
- fps
- width
- height
- codec
- created_at
- processed_at
- config_hash
- status

### frame_metrics

Suggested columns:

- video_id
- frame_index
- timestamp_ms
- variance
- sharpness
- motion
- edge_density
- exposure_mean
- exposure_p95
- exposure_p99
- glare_ratio
- presence_score
- selected_for_detection
- window_id
- metadata_json

### presence_windows

Suggested columns:

- window_id
- video_id
- start_frame_index
- end_frame_index
- start_timestamp_ms
- end_timestamp_ms
- confidence
- source
- selected_frame_indices_json
- metadata_json

### detections

Extend existing table with:

- detection_id
- video_id
- frame_index
- timestamp_ms
- window_id
- polygon_json
- bbox_json
- confidence
- label
- detector_name
- detector_version
- geometry_valid
- rejection_reason
- metadata_json

### card_instances

New table:

- instance_id
- video_id
- start_timestamp_ms
- end_timestamp_ms
- start_frame_index
- end_frame_index
- side
- status
- canonical_view_id
- canonical_image_path
- fused_display_image_path
- detection_count
- view_count
- metadata_json

### card_views

New table:

- view_id
- instance_id
- video_id
- frame_index
- timestamp_ms
- detection_id
- window_id
- raw_frame_path
- raw_crop_path
- rectified_raw_path
- normalized_path
- polygon_json
- homography_json
- canonical_width
- canonical_height
- orientation_degrees
- sharpness
- exposure_mean
- exposure_p95
- glare_ratio
- glare_centroid_x
- glare_centroid_y
- pose_features_json
- lighting_features_json
- pose_cluster
- lighting_cluster
- representative_role
- metadata_json

### multi_frame_composites

New table:

- composite_id
- instance_id
- source_view_ids_json
- output_path
- method
- alignment_success
- alignment_error
- metadata_json

---

## 7. Configuration

Use YAML configuration for major pipeline behavior.

Example:

```yaml
pipeline:
  version: 2
  mode: full
  save_diagnostics: true
  save_raw_frames: true
  save_rejected_frames: false

video_decode:
  strategy: sequential
  scan_fps: 5
  lowres_width: 160
  candidate_fullres: true

sampler:
  type: adaptive_contrast_window
  threshold_mode: median_mad
  threshold_k: 2.5
  min_window_ms: 700
  max_window_ms: 5000
  merge_gap_ms: 300
  candidates_per_window: 8
  fusion_candidates_per_window: 12
  evidence_candidates_per_window: 12
  motion_as_positive_signal: false
  edge_density_as_positive_signal: false

metrics:
  compute_variance: true
  compute_sharpness: true
  compute_motion: true
  compute_edge_density: true
  compute_exposure: true
  compute_glare: true

model:
  detector_type: yolo
  model_path: models/card_detector.pt
  confidence_threshold: 0.25
  batch_size: auto
  max_batch_size: 32
  device: auto

geometry:
  corner_refinement: true
  fallback_to_bbox: true
  min_card_area_ratio: 0.05
  max_card_area_ratio: 0.95
  expected_aspect_ratio: 0.7142857
  aspect_ratio_tolerance: 0.25

canonical_image:
  enabled: true
  width: 1125
  height: 1575
  orientation: portrait
  save_rectified_raw: true
  save_normalized: true

normalization:
  enabled: true
  white_balance: true
  exposure_normalization: true
  contrast_normalization: mild
  aggressive_glare_removal: false
  generative_enhancement: false

instance_grouping:
  enabled: true
  use_presence_windows: true
  use_perceptual_hash: true
  hash_method: phash
  hash_distance_threshold: 8
  max_time_gap_ms: 1500

representative_views:
  enabled: true
  max_views_per_instance: 8
  roles:
    - canonical
    - low_glare
    - specular
    - left_tilt
    - right_tilt
    - high_detail
    - raw_evidence

fusion:
  enabled: true
  method: median
  max_source_views: 8
  require_similar_pose: true
  save_source_view_ids: true

review_ui:
  show_instances: true
  show_evidence_views: true
  show_diagnostics: true
```

---

## 8. Module Structure

Suggested v2 code organization:

```text
card_capture/
  __init__.py
  cli.py
  config.py
  models.py
  pipeline.py

  video/
    decode.py
    metrics.py
    timeline.py

  sampling/
    windows.py
    candidate_selection.py

  detection/
    base.py
    yolo.py
    fake.py

  geometry/
    corners.py
    validation.py
    rectification.py
    orientation.py

  instances/
    grouping.py
    hashing.py
    representative_views.py

  enhancement/
    normalization.py
    fusion.py

  diagnostics/
    writer.py
    plots.py
    rejection_reasons.py

  storage/
    sqlite.py
    filesystem.py
    schema.py

  review/
    app.py
    templates/
```

The current modules can be migrated incrementally.

---

## 9. Implementation Plan

## Phase 1: Data Model and Storage Foundation

### Goals

Introduce v2 concepts without changing all processing behavior at once.

### Tasks

- Add `CardInstance` model.
- Add `CardView` model.
- Add `FrameMetric` model if not already present.
- Add `CardPresenceWindow` model.
- Add database migrations or schema updates.
- Add file layout helpers.
- Add config loader.
- Ensure existing pipeline can still run.

### Acceptance Criteria

- Existing MVP behavior still works.
- New tables can be created.
- New objects can be serialized/deserialized.
- Processing config is saved with each run.

---

## Phase 2: Diagnostics and Metric Collection

### Goals

Collect enough information to understand misses and train future models.

### Tasks

- Write `frame_metrics.csv` for every run.
- Write `windows.json`.
- Write `detections.json`.
- Write `card_instances.json` once instances exist.
- Log rejection reasons.
- Add optional metric timeline plot.

### Acceptance Criteria

For a processed video, developer can answer:

- Did the sampler detect a card-present window?
- Which frames were selected for detection?
- Did the detector fire?
- Did geometry validation fail?
- Which views were selected as evidence?

---

## Phase 3: Sequential Decode and Adaptive Windows

### Goals

Reduce video I/O bottlenecks and make card presence detection more robust.

### Tasks

- Implement sequential metric scan.
- Replace fixed contrast threshold with adaptive threshold.
- Disable motion OR-trigger by default.
- Make edge density collection diagnostic-only by default.
- Emit `CardPresenceWindow` objects.

### Acceptance Criteria

- 60-second video does not require repeated expensive frame seeking.
- Presence windows are visible in diagnostics.
- Threshold values are recorded per video.

---

## Phase 4: Batch Detection

### Goals

Improve inference throughput.

### Tasks

- Add `detect_batch` interface.
- Implement YOLO batch inference.
- Add adaptive batch sizing.
- Record detector metadata.

### Acceptance Criteria

- Batch detector returns equivalent detection structure to single-frame detector.
- Pipeline can switch between single and batch detection.
- Batch size can be configured or set to auto.

---

## Phase 5: Geometry and Canonical Images

### Goals

Produce consistent vertical card images.

### Tasks

- Implement corner ordering.
- Implement homography estimation.
- Implement perspective warp to fixed canvas.
- Implement portrait orientation normalization.
- Save rectified raw image per CardView.
- Save normalized review image per CardView.

### Acceptance Criteria

- Each valid detection produces a portrait card image with consistent dimensions.
- Raw evidence is preserved separately.
- Homography and polygon metadata are stored.

---

## Phase 6: Card Instance Grouping

### Goals

Group multiple views of the same physical card side.

### Tasks

- Group views by presence window and time.
- Add perceptual hashing on rectified crops.
- Avoid duplicate saved outputs.
- Create CardInstance records.
- Assign CardViews to instances.

### Acceptance Criteria

- A single held card produces one CardInstance with multiple CardViews.
- Duplicate frames are grouped rather than saved as separate cards.
- Front/back or distinct cards are not aggressively merged.

---

## Phase 7: Representative Evidence Views

### Goals

Collect diverse views for future grading without implementing grading.

### Tasks

- Compute pose features.
- Compute lighting/glare features.
- Cluster views by pose and lighting.
- Select representative roles.
- Save role-labeled evidence views.

### Acceptance Criteria

Each CardInstance attempts to produce:

- canonical view
- low-glare view
- specular view if present
- angle-diverse views if present
- raw evidence view

Missing roles are allowed if not present in the video.

---

## Phase 8: Multi-Frame Fusion

### Goals

Generate optional clean display images from several aligned frames.

### Tasks

- Select fusion candidates.
- Align in canonical card space.
- Implement median or weighted average fusion.
- Save fused display image.
- Store source view IDs.

### Acceptance Criteria

- Fused image is saved as display asset.
- Source views are retained.
- Fusion failure does not fail the whole pipeline.

---

## Phase 9: Review UI Updates

### Goals

Make the new data model usable by humans.

### Tasks

- Show CardInstances instead of only individual detections.
- Show canonical image by default.
- Allow expanding instance to view evidence set.
- Show raw, rectified, normalized, and fused variants.
- Show diagnostics per card instance.
- Allow manual front/back labeling if desired.

### Acceptance Criteria

- User can review one card instance at a time.
- User can inspect evidence views.
- User can approve/reject card instance outputs.

---

## 10. Future Defect and Grading Model Readiness

No grading scores should be implemented in this phase.

However, the following data should be collected to support future grading:

### Centering

Needed later:

- rectified card image
- border segmentation or card layout detection
- canonical dimensions

Collect now:

- rectified raw card image
- normalized card image
- homography
- polygon

### Corner Wear

Needed later:

- high-resolution corner crops
- raw and normalized views

Collect now:

- canonical card views
- evidence view set
- optional future corner crop export

### Edge Wear

Needed later:

- edge strips from canonical card
- lighting-diverse evidence

Collect now:

- rectified raw views
- pose/lighting metadata

### Surface Scratches

Needed later:

- multi-angle views
- specular highlights
- raw evidence

Collect now:

- glare maps
- specular representative views
- pose/lighting clusters

### Dents / Indentations

Needed later:

- raking light or changing reflections
- multiple lighting angles

Collect now:

- specular movement metadata
- frames with different glare centroids
- angle-diverse evidence

---

## 11. Testing Strategy

## 11.1 Unit Tests

Add tests for:

- config parsing
- adaptive threshold calculation
- window grouping
- candidate selection
- corner ordering
- homography generation
- perspective rectification dimensions
- perceptual hashing
- instance grouping
- representative view selection
- fusion fallback behavior
- database serialization

## 11.2 Integration Tests

Use short fixture videos or synthetic frame sequences.

Test cases:

1. one card held steady
2. one card tilted through multiple angles
3. two cards shown close together in time
4. low-contrast card
5. glare-heavy card
6. camera movement without card change
7. card front and back
8. false positive background/lightbox edge

## 11.3 Golden Output Tests

For known videos, assert:

- expected number of CardInstances
- minimum number of CardViews per instance
- canonical image dimensions
- diagnostics files exist
- no duplicate instances for one held card

## 11.4 Performance Tests

Track:

- total processing time
- decode time
- metric computation time
- detector time
- rectification time
- fusion time
- storage time

Add simple regression thresholds.

---

## 12. Metrics to Track

Even before grading/scoring, track system-level metrics.

### Detection / Extraction Metrics

- card instance recall on labeled test videos
- false positive instance count
- duplicate instance count
- windows detected per video
- views per instance
- evidence roles filled per instance

### Performance Metrics

- seconds per input video minute
- frames decoded per second
- detector frames per second
- batch size used
- GPU/CPU device used

### Data Quality Metrics

- rectification success rate
- geometry validation failure rate
- average sharpness by view role
- average glare ratio by view role
- fusion success rate

---

## 13. LLM Implementation Guidance

When using an LLM to implement this specification, prefer small, incremental changes.

Recommended implementation order for LLM coding sessions:

1. Add dataclasses and schema migrations.
2. Add configuration file and config parsing.
3. Add diagnostics writer.
4. Add sequential metric scan.
5. Add adaptive window detection.
6. Add batch detector interface.
7. Add perspective rectification.
8. Add CardView persistence.
9. Add CardInstance grouping.
10. Add representative evidence selection.
11. Add optional fusion.
12. Update review UI.

Each coding session should include:

- one focused feature
- unit tests
- backward compatibility check
- diagnostic output validation

Avoid combining data model, video decode, detection, and UI changes in a single implementation step.

---

## 14. Non-Goals for This Version

The following should not be implemented yet:

- final grading scores
- automated PSA/BGS/SGC grade prediction
- defect severity scoring
- generative image repair
- cloud processing
- card identity recognition unless needed for grouping
- pricing or marketplace lookup

These can be added later once the v2 evidence and metadata pipeline is stable.

---

## 15. Summary of Required v2 Features

This v2 architecture should support:

- sequential video processing
- adaptive card presence windows
- batch detection
- corner refinement
- perspective-corrected vertical card images
- raw evidence preservation
- normalized review images
- CardInstance grouping
- multiple CardViews per instance
- multi-angle / lighting-diverse evidence collection
- optional multi-frame fused display image
- diagnostic artifacts
- future grading model readiness
- config-driven experimentation

The central architectural shift is:

```text
from: selecting one best frame per detected card

to: building a structured multi-view evidence package per card instance
```

That shift should make the system more accurate now and much easier to extend later.

