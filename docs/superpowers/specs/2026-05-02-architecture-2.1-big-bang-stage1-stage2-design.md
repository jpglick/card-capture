# Architecture 2.1 Big-Bang Replacement (Stage 1 + Stage 2)

## Status
Approved for planning and implementation.

## Scope
This spec defines a big-bang replacement of the current card-capture runtime for:

1. Stage 1: Ingestion and fast triage
2. Stage 2: Zero-shot corner detection

The design intentionally includes a breaking database migration and removes legacy runtime paths from the main process flow.

## Goals

1. Replace `cv2.VideoCapture` ingestion with `decord` as primary reader and `PyAV` fallback.
2. Replace YOLO box detection path with ONNXRuntime-first corner detection abstractions.
3. Replace synchronous single-thread processing with producer/consumer multiprocessing.
4. Replace current persistence contract with v2.1 entities (`CardInstance`, `CardView`, `Evidence` aligned schema).
5. Preserve strict typed contracts between pipeline stages.

## Non-Goals (This Cut)

1. Full Stage 3 object tracking and multi-frame grouping logic.
2. Stage 4 homography rectification output generation.
3. Stage 5 quality scoring and canonical view selection.
4. Backward-compatible dual schema writes or compatibility views.

## Architecture Overview

### Runtime Topology

The pipeline is executed as a multiprocessing producer/consumer system:

1. Producer process:
   - Reads frames sequentially from `decord.VideoReader`.
   - Falls back to PyAV when Decord is unavailable.
   - Computes lightweight triage metrics (blur/variance and empty-frame checks).
   - Pushes accepted `FramePacket` objects to a bounded multiprocessing queue.

2. Consumer process:
   - Pulls `FramePacket` items in batches from the queue.
   - Runs ONNX corner detection via detector adapter.
   - Emits `DetectionPacket` objects with four ordered corners, confidence, and metadata.

3. Main/orchestrator process:
   - Owns process startup/shutdown and queue lifecycle.
   - Handles poison-pill signaling and failure propagation.
   - Persists v2.1 entities through the rewritten storage layer.

### Error and Shutdown Semantics

1. Producer pushes a sentinel when EOF is reached.
2. Consumer drains remaining frames, flushes pending batch inference, then exits cleanly.
3. Child exceptions are propagated to parent and mark the run as failed.
4. Parent process guarantees queue/process cleanup regardless of failure location.

## Component Contracts and Module Restructure

### `models.py`

Add/replace data contracts with strict dataclasses and type hints:

- `FramePacket`
  - `frame_index: int`
  - `timestamp_ms: int`
  - `image: np.ndarray`
  - `width: int`
  - `height: int`
  - `triage_metrics: dict[str, float]`

- `CornerDetection`
  - `corners: tuple[Point, Point, Point, Point]` in TL/TR/BR/BL order
  - `confidence: float`
  - `metadata: dict[str, Any]`

- `DetectionPacket`
  - Frame identity fields
  - `corner_detection: CornerDetection`

- `CardView`, `CardInstance`, `EvidenceRef`
  - Present in this cut with fields needed for Stage 1/2 persistence.
  - Stage 3+ fields remain nullable or optional where not yet computed.

### `ingestion.py` (new)

Replace cv2 sampler-centric ingestion flow with:

- `FrameReaderProtocol`
- `DecordFrameReader`
- `PyAvFrameReader`
- `FrameTriageFilter`

`FrameTriageFilter` owns deterministic pass/drop behavior based on configured blur/variance and empty-frame thresholds.

### `detectors.py`

Replace YOLO-first detector logic with corner detector abstractions:

- `CornerDetector` protocol
- `OnnxCornerDetector` base adapter
- `DocAlignerCornerDetector` concrete ONNX implementation
- `FakeCornerDetector` test adapter

The concrete adapter normalizes corner ordering to TL/TR/BR/BL and enforces confidence thresholding.

### `pipeline.py`

Replace synchronous frame loop with multiprocessing orchestration:

- producer target function
- consumer target function
- queue setup (bounded size)
- poison-pill handling
- child exception propagation
- consumer-side configurable inference batch size

Legacy sampler + YOLO orchestration code is removed from active processing path.

### `storage.py`

Rewrite for v2.1 schema only. Storage methods align to:

- run/video creation
- card instance insertion
- card view insertion
- evidence frame insertion

No legacy-table compatibility methods are retained in runtime paths.

### `cli.py`

Replace legacy detector/sampler controls with v2.1 parameters:

- reader backend preference (`auto`, `decord`, `pyav`)
- queue size
- inference batch size
- corner confidence threshold
- triage thresholds

CLI validation must fail fast on invalid backend/model configuration.

## Data Model and Breaking Migration

### Target Tables

1. `videos` (retained with updated metadata usage)
2. `card_instances`
   - `id`, `video_id`, `track_id`, `created_at`, `updated_at`
3. `card_views`
   - `id`, `card_instance_id`, `frame_index`, `timestamp_ms`
   - `corners_json`, `confidence`
   - `rectified_path` (nullable until Stage 4)
   - `quality_score_json` (nullable until Stage 5)
   - `is_canonical` (default false)
4. `evidence_frames`
   - `id`, `card_view_id`, `source_frame_path`, `frame_width`, `frame_height`
   - optional metrics JSON
5. `pipeline_runs` (recommended observability table)
   - backend/model/threshold summary and status fields

### Stage 1/2 Persistence Behavior

1. Each accepted Stage 2 detection is persisted as a provisional `card_instance`.
2. `card_view` rows store frame/corner/confidence metadata.
3. Raw frame artifacts are stored and linked via `evidence_frames`.
4. `rectified_path` and `quality_score_json` remain null in this cut.

### Migration Policy

1. This is a breaking migration.
2. `Storage.initialize()` creates/applies v2.1 tables only.
3. Legacy tables/flows are removed from active logic.
4. Existing tests/UI depending on legacy tables are updated or explicitly gated until Stage 4/5 equivalents exist.

## Testing and Verification Plan

### Unit Tests

1. Ingestion:
   - Decord reader frame ordering
   - PyAV fallback behavior
   - triage pass/drop determinism
2. Detector adapters:
   - ONNX input/output shape validation
   - corner ordering normalization
   - confidence threshold behavior
3. Multiprocessing pipeline:
   - producer/consumer queue handoff
   - sentinel termination
   - child exception propagation

### Storage Tests

1. v2.1 schema bootstrap
2. insert/read for `card_instances`, `card_views`, `evidence_frames`
3. nullability behavior for pre-Stage-4/5 fields

### Integration Test

1. Synthetic short video fixture
2. Fake corner detector
3. Full process command run
4. Assertions:
   - nonzero detections
   - expected row counts in v2.1 tables
   - evidence files written

### CLI Tests

1. new v2.1 flag parsing
2. invalid backend/model error behavior
3. `auto` backend selection path

### Acceptance Gate

Before marking this cut complete:

1. relevant pytest targets pass
2. local real-video smoke run passes end-to-end
3. v2.1 DB rows and artifact links verified
4. no active runtime dependency on legacy YOLO/sampler flow

## Risks and Mitigations

1. Dependency instability (`decord` availability):
   - Mitigation: first-class `PyAV` fallback with identical frame packet contract.
2. Multiprocessing failure visibility:
   - Mitigation: explicit exception channels and parent-side fail-fast behavior.
3. Big-bang regression risk:
   - Mitigation: strict contract tests, integration test with fake detector, smoke test with real video.

## Rollout Notes

1. Documentation is updated concurrently with code changes.
2. Existing output directories may need cleanup when switching to new artifact conventions.
3. Review UI may remain temporarily partial until Stage 4/5 populate rectified/canonical fields.
