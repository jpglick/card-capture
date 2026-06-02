# Card Capture — Agent Context

Extract high-quality 750×1050 stills of trading cards from hand-held workspace
videos. Input: 4K portrait `.MOV` files. Output: perspective-rectified,
deduplicated card images + SQLite metadata.

Full v5.5 architecture: `docs/architecture/arch-5.5.md`.

---

## Architecture (Local Runtime, v5.5+)

The pipeline runs as a high-performance in-process loop (`LocalPipelineRuntime`) to
minimize IPC overhead and redundant video decoding.

```
Video.mov
  Stage 1  sample     — Adaptive Presence Sampler; starts streaming producer
  Stage 2  detect     — YOLO Corner Detection; batched inference
  Stage 3  novelty    — Background Novelty Gate; drops empty stands
  Stage 4  track      — Session-Aware Tracking; BoT-SORT/ByteTrack
  Stage 5  refine     — GPU Refinement; Kornia perspective warp 750x1050
  Stage 6  score      — Quality Scoring + Pruning; adaptive thresholds
  Stage 7  resolve    — Front/Back Resolution; side prediction
  Stage 8  fuse       — Lighting-Diverse Fusion; median fusion
  Stage 9  dedup      — Global Dedup; ReID + pHash
  Stage 10 store      — Storage; persists to disk and SQLite
```

**Architectural Mandates:**
- **In-Process:** Production runs must use `LocalPipelineRuntime`.
- **Strict GPU Boundary:** All PyTorch/Kornia operations MUST happen in the `_worker` thread context.
- **Single-Writer DAL:** All DB writes MUST go through the `Writer` in `card_capture.data.writer`.

---

## Key Modules

```
src/card_capture/
  core/         # foundation layer — leaf utilities + types (models, config, gpu_utils)
  stages/       # vertical slices (sample, detect, novelty, track, refine, score, resolve, fuse, dedup, store)
  shared/       # cross-stage helpers (pipeline_utils, stage_metrics)
  ml/           # model zoo + inference (shared assets)
  training/     # all offline training logic
  runtime/      # GPU session orchestration (gpu_session, strict_gpu)
  pipeline/     # orchestration (request, runner, telemetry, runtime_local)
  data/         # DAL (connection, repositories)
  review/       # legacy Jinja review UI
```

---

## Configuration

All knobs live in `PipelineConfig` (`src/card_capture/pipeline/request.py`). Defaults:

```
corner_confidence          = 0.5    # YOLO gate
background_novelty_threshold = 0.08 # empty-workspace gate
fast_scan_fps              = 15.0   # sampler scan speed
valley_drop_ratio          = 0.40   # valley sensitivity for card swaps
tracker_backend            = "bytetrack"
min_track_length           = 3
fusion_target_frames       = 1
rotate_180                 = False  # flip for upside-down camera
```

---

## Testing & Baseline

```bash
# All unit tests (excluding quarantined hardware-dependent tests)
python3 -m pytest tests/ -m "not quarantine" -q

# Established v5.5 baseline results
# Location: docs/superpowers/plans/v5-5/baseline-results.md
```

Quarantined tests (`@pytest.mark.quarantine`) include those requiring CUDA/MPS hardware or missing external credentials.

---

## Commands

```bash
# Process a video (Local pipeline, default)
card-capture process video.MOV --output-dir out --db out/cards.sqlite

# Start the web app
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
# UI: http://localhost:8000 (FastAPI handles static files)
```

---

## Known Weaknesses (v5.5)

- F/B classifier fallback uses longest-track heuristic when the classifier is unavailable.
- GPU Refinement uses CPU fallback (`PrecisionNormalizer`) if Kornia construction fails for the requested device.
- Apple Silicon (MPS) is the only supported hardware accelerator; CUDA is unsupported.
