# Card Capture — Agent Context

Extract high-quality 750×1050 stills of trading cards from hand-held workspace
videos. Input: 4K portrait `.MOV` files. Output: perspective-rectified,
deduplicated card images + SQLite metadata.

---

## Pipeline (10 stages, Metaflow)

```
Video.mov
  Stage 1  Adaptive Presence Sampler   — 192px fast scan @ 15fps; valley-split detection
  Stage 2  Frame Triage                — drop >98% dark frames
  Stage 3  YOLO Corner Detection       — YOLOv8-OBB @ 640px, conf ≥ 0.5, batched GPU
  Stage 4  Background Novelty Gate     — drop frames matching empty workspace
  Stage 5  Session-Aware Tracking      — BoT-SORT or ByteTrack; 4 reset signals
  Stage 6  GPU Refinement              — Kornia perspective warp → 750×1050
  Stage 7  Quality Scoring             — 7-component score; prune weak tracks
  Stage 8  Front/Back Resolution       — longest track = Front; pHash Back gate
  Stage 9  Lighting-Diverse Fusion     — 4-quadrant glare selection; median or foil-aware fuse
  Stage 10 Global Dedup + Storage      — pHash + ReID; write cards.sqlite
```

Stages 1–3: producer/consumer subprocesses.  
Stages 4–10: main process (Metaflow steps in `pipeline/steps/`).  
Monolith path (`src/card_capture/pipeline.py`) is deprecated; deleted in Wave 5.

---

## Key Modules

| File | Purpose |
|---|---|
| `pipeline/card_capture_flow.py` | Metaflow orchestrator — thin `@step` calls into `pipeline/steps/` |
| `pipeline/steps/` | One module per stage (detect, novelty, track, refine, score, resolve, fuse, dedup, store) |
| `src/card_capture/sampler/__init__.py` | Stage 1: two-pass presence sampler + valley splits |
| `src/card_capture/detectors.py` | Stage 3: YOLOv8-OBB inference |
| `src/card_capture/presence/background_novelty.py` | Stage 4: novelty gate |
| `src/card_capture/tracking/botsort_adapter.py` | Stage 5: BoT-SORT + ReID |
| `src/card_capture/tracking/bytetrack_adapter.py` | Stage 5: ByteTrack fallback |
| `src/card_capture/tracking/centroid_jump.py` | Session reset signal #3 |
| `src/card_capture/gpu_refinement.py` | Stage 6: Kornia warp (GPU) |
| `src/card_capture/cropper.py` | Stage 6: CPU fallback |
| `src/card_capture/scoring.py` | Stage 7: quality components |
| `src/card_capture/fuser.py` | Stage 9: frame selection + fusion |
| `src/card_capture/fusion/foil_detection.py` | Laplacian variance foil classifier |
| `src/card_capture/fusion/median_fusion.py` | Glare-rejection fusion for foil cards |
| `src/card_capture/deduplicator.py` | Stage 10: pHash + cosine dedup |
| `src/card_capture/storage.py` | SQLite schema + persistence |
| `src/card_capture/config.py` | `PipelineConfig` dataclass + `load_config()` |
| `src/card_capture/adaptive_gap.py` | Per-video session-split gap computation |
| `src/card_capture/cli.py` | CLI entry point (`card-capture` command) |
| `app/main.py` | FastAPI app factory |
| `app/api/` | REST routes (videos, runs, cards, label, training, regression, config) |
| `app/services/` | Domain services used by routes |
| `app/web/src/` | Svelte SPA |
| `migrations/run_migrations.py` | SQLite migration runner + startup assertion |
| `harness/` | Regression harness (metrics, runner, CLI) |

---

## Configuration

All knobs live in `PipelineConfig` (`src/card_capture/config.py`). Defaults:

```
corner_confidence          = 0.5    # YOLO gate
background_novelty_threshold = 0.08 # empty-workspace gate
fast_scan_fps              = 15.0   # sampler scan speed
valley_drop_ratio          = 0.40   # valley sensitivity for card swaps
centroid_jump_ratio        = 0.30   # tracking reset on position jump
foil_threshold             = 50.0   # Laplacian variance; 0 = always median
rotate_180                 = False  # flip for upside-down camera
tracker_backend            = "bytetrack"
min_track_length           = 6
reid_distance_threshold    = 0.6
fusion_target_frames       = 4
```

Config file: `card_capture_config.json` (gitignored; copy from `harness/config.example.json`).  
CLI flags override config. Presets (fast/balanced/quality) live in `cards.sqlite:config_presets`.

---

## Key Data Types

```python
FrameSample(frame_index, timestamp_ms, image, w, h)
DetectionPacket(frame_index, timestamp_ms, w, h, corner_detection, telemetry)
ScoredCandidate(detection_id, timestamp_ms, image_path, score, corners, frame_index)
TrackState(instance_id, candidates, last_centroid, last_frame_index, angle, reid_embedding)
QualityScore(sharpness, glare, aspect_ratio, size, complexity, border_purity, confidence, total)
```

Quality score weights: sharpness 25%, border_purity 20%, aspect_ratio 15%, glare 15%,
complexity 10%, size 10%, confidence 5%.

---

## Session Reset Signals (Stage 5)

A new tracking session is started when any of these fire:
1. Frame-index gap (sampler window boundary)
2. Valley split (hand-swap detected in Stage 1)
3. Centroid jump (> 0.30× frame width)
4. ReID shift (BoT-SORT only)

---

## Testing

```bash
# All unit tests
python3 -m pytest tests/ -q

# Skip the slow integration test (requires fixture video)
python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py
```

Pre-existing failures (not regressions): `tests/migrations/test_schema.py::test_migrations_are_idempotent`,
several in `test_wave1/2_robustness.py`, `test_path_equivalence.py`.

---

## Commands

```bash
# Process a video (Metaflow pipeline, default)
card-capture process video.MOV --output-dir out --db out/cards.sqlite

# Stage 1 only — fast sanity check (~35s)
card-capture sampler sessions video.MOV

# Start the web app (two terminals)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload   # terminal 1
cd app/web && npm run dev                                      # terminal 2
# → http://localhost:5173

# Regression harness
card-capture harness run --baseline v1 --db cards.sqlite --truth-dir golden_set/

# Export presence training data
card-capture dataset export --db cards.sqlite --out-dir data/presence_dataset

# Train presence classifier
card-capture train presence --data data/presence_dataset --out models/presence_classifier.pt
```

---

## Output Structure

```
<output_dir>/
  frames/                     source frames (one per detection)
  crops/                      fused canonical images (750×1050 px)
  cards.sqlite                all metadata, events, embeddings
  run_telemetry.json
  tracker_association_events.json
```

Database tables: `videos`, `card_instances`, `card_views`, `pipeline_events`,
`config_presets`, `fb_labels`, `truth_files`, `regression_baselines`, `_migrations`.

---

## Known Weaknesses (see `docs/V4_CONCERNS.md`)

- Background novelty gate uses single mean (no variance model)
- pHash Front/Back gate uses rotation tolerance as proxy for content similarity
- BoT-SORT ReID fed real fused images now (late embedding in `store.py`) but tracker path still degraded
- Quality scorer is hand-weighted; no learned ranker
- Settings UI exposes 5 of ~30 tunable thresholds

---

## Docs

- `docs/architecture/arch-4.1.md` — detailed v4.1 spec
- `docs/architecture/roadmap.md` — phase plan, what's shipped vs pending
- `docs/V4_CONCERNS.md` + `docs/V4_CONCERNS_PASS2.md` — open issues and review findings
- `OPERATOR.md` — running the app, processing videos, training walkthrough
- `QUICK_REFERENCE.md` — CLI flags cheat sheet
