# Card Capture — Agent Context

Extract high-quality 750×1050 stills of trading cards from hand-held workspace
videos. Input: 4K portrait `.MOV` files. Output: perspective-rectified,
deduplicated card images + SQLite metadata.

---

## Architecture (Unified Runtime, v5.5+)

The pipeline runs as a high-performance in-process loop (`UnifiedRuntime`) to
minimize IPC overhead and redundant video decoding.

```
Video.mov
  [Producer Thread]
  Stage 1  Stride Sampler             — Decodes frames @ variable FPS based on activity
  
  [Worker Thread (GPU Boundary)]
  Stage 2  YOLO Corner Detection       — YOLOv8-OBB @ 640px; batched GPU inference
  Stage 3  Eager Warp / Crop Cache    — GPU-resident 750×1050 crops; skips re-decoding
  
  [Main Thread]
  Stage 4  Background Novelty Gate     — Mean + Variance model; drops empty stands
  Stage 5  Session-Aware Tracking      — BoT-SORT/ByteTrack; 4 reset signals
  Stage 6  Refinement & Scoring        — Consumes cached crops; 7-component quality score
  Stage 7  Front/Back Resolution       — Heuristic or classifier-based resolution
  Stage 8  Lighting-Diverse Fusion     — Median or foil-aware glare rejection
  Stage 9  Global Dedup + Storage      — Single-Writer DAL; cards.sqlite
```

**Architectural Mandates:**
- **In-Process:** Production runs must use `UnifiedRuntime`. Metaflow is for remote orchestration only.
- **Strict GPU Boundary:** All PyTorch/Kornia operations MUST happen in the `_worker` thread.
- **Single-Writer DAL:** All DB writes MUST go through `SingleWriterDAL` in `card_capture.dal`.

---

## Key Modules

| File | Purpose |
|---|---|
| `src/card_capture/runtime.py` | **Core:** Unified in-process orchestrator and producer/worker threads |
| `src/card_capture/dal.py` | Data Access Layer: Single-Writer thread-safe SQLite persistence |
| `src/card_capture/models.py` | Centralized domain objects (`FrameSample`, `TrackState`, etc.) |
| `src/card_capture/interfaces.py` | Protocols for components (`CardDetector`, `FrameSampler`) |
| `src/card_capture/sampler/` | `StrideSampler` for two-pass presence detection |
| `src/card_capture/detectors.py` | YOLOv8-OBB backends + device probing |
| `src/card_capture/presence/` | Novelty gate (Mean/Variance) + background modeling |
| `src/card_capture/tracking/` | BoT-SORT and ByteTrack adapters |
| `src/card_capture/cropper.py` | `PrecisionNormalizer` for consistent homography |
| `src/card_capture/storage.py` | (Internal) Direct SQLite storage logic |
| `pipeline/card_capture_flow.py` | Metaflow orchestrator (Remote/Baseline use only) |

---

## Configuration

All knobs live in `PipelineConfig` (`src/card_capture/config.py`). Defaults:

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
# Process a video (Unified pipeline, default)
card-capture process video.MOV --output-dir out --db out/cards.sqlite

# Run via Metaflow (legacy/remote)
card-capture process video.MOV --pipeline metaflow --db out/cards.sqlite

# Start the web app
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
# UI: http://localhost:8000 (FastAPI handles static files)
```

---

## Known Weaknesses (v5.5)

- F/B classifier fallback uses longest-track heuristic when the classifier is unavailable (resolve stage gracefully degrades; classifier weights are an optional artifact).
- Eager warping uses CPU fallback (`PrecisionNormalizer`) if Kornia construction fails for the requested device.
- In-memory peak (~180 MB for the reference video) scales with concurrent active tracks; mitigation (selective spill between refine/score) is a tracked follow-up.

## Recent baseline

V5.5 back-half wired and verified against IMG_5872.MOV — see
[docs/superpowers/plans/v5-5/back-half-baseline.md](docs/superpowers/plans/v5-5/back-half-baseline.md).
