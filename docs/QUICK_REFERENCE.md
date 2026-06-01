# Quick Reference: Card Capture v5.5

## v5.5 Problem Statement
Extract high-quality trading card stills from local videos with an in-process, Apple-Silicon-optimized pipeline.

## High-Level v5.5 Flow
```
Video -> sample (Stage 1) -> detect (Stage 2) -> novelty (Stage 3)
      -> track (Stage 4) -> refine (Stage 5) -> score (Stage 6)
      -> resolve (Stage 7) -> fuse (Stage 8) -> dedup (Stage 9)
      -> store (Stage 10)
```

This is an in-process producer/consumer pipeline implemented in `src/card_capture/pipeline/runtime_local.py`.

## v5.5 CLI Flags (process)

```bash
card-capture process <video> \
  --output-dir <dir> \
  --db <db.sqlite> \
  --detector {docaligner,fake} \
  --reader-backend {auto,decord,pyav} \
  --corner-confidence 0.5 \
  --detection-width 640 \
  --device {auto,cpu,mps}
```

## Install Notes

- `pip install -e ".[legacy_tracking]"` installs the tracking dependencies: `av` and `onnxruntime`.
- `decord` is a separate install because PyPI does not ship Apple Silicon macOS wheels.
- `--reader-backend auto` prefers `decord` when importable and otherwise falls back to `pyav`.
- Apple Silicon macOS: use a local micromamba/conda-forge environment.

## v5.5 Runtime Entities

- `PipelineRunRequest`: serializable request with config knobs.
- `LocalPipelineRuntime`: orchestrator for the in-process run.
- `RunManifest`: final result with timings and card records.
- `PipelineRunResult`: top-level result containing the manifest.

## Stage Responsibilities

1. **sample**: Streaming decode producer.
2. **detect**: YOLOv8-OBB corner detection (batched).
3. **novelty**: Background novelty gating.
4. **track**: BoT-SORT/ByteTrack session-aware tracking.
5. **refine**: Kornia GPU perspective warp to 750×1050.
6. **score**: Quality scoring and adaptive pruning.
7. **resolve**: Front/Back and identity resolution.
8. **fuse**: Median/Foil-aware glare rejection.
9. **dedup**: Global intra-run and cross-video deduplication.
10. **store**: Metadata persistence and image writing via repositories.

## Performance Mandate

- **In-Process:** Avoids IPC overhead.
- **GPU Boundary:** All model inference and GPU ops are confined to a worker context.
- **Single-Writer:** All SQLite writes are sequentialized through `card_capture.data.writer`.

## Smoke Command (v5.5)

```bash
card-capture process <temp-video> \
  --output-dir <temp-out> \
  --db <temp-db> \
  --detector fake \
  --reader-backend auto \
  --corner-confidence 0.5
```
