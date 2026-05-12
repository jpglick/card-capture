# Card Capture: Video Processing Pipeline Context

**Last Updated:** May 2026  
**Current Branch:** `claude/improve-video-pipeline-2wsxP`  
**Latest Work:** Wave 3/4 foil-aware fusion, glare rejection, adaptive calibration

---

## 1. Problem & Goals

Extract high-quality, normalized 750×1050 stills of trading cards from hand-held workspace videos. A user presents cards to camera (often hand-held), flips to show back, removes them. The pipeline must:

- **Locate** sharpest frames where card is fully visible and not moving
- **Detect** 4 corners of card regardless of orientation
- **Rectify** perspective to perfect flat 2.5×3.5 ratio
- **Group** Front/Back views of same physical card
- **Deduplicate** identical cards across videos

### Input Characteristics
- **Resolution:** 4K+ (e.g., 2160×3840), portrait mode, 30-60 fps
- **Scene:** Workspace (desk/stand) with continuous video capture
- **Challenges:** Motion blur, glare, occlusions (fingers), perspective skew, foil/holographic shimmer, empty-stand transitions

---

## 2. The 10-Stage Pipeline Architecture

```
Video.mov
    ├─ Stage 1: Adaptive Presence Sampler (subprocess, two-pass)
    │  └─ fast scan (192px @ 15fps) + sharpness extraction
    │  └─ outputs: PresenceWindows, valley_splits, background_proxies
    ├─ Stage 2: Frame Triage Filter (producer subprocess)
    │  └─ drop empty frames (>98% dark pixels)
    ├─ Stage 3: YOLO Corner Detection (consumer subprocess, batched GPU)
    │  └─ YOLOv8-OBB @ 640px input, conf threshold 0.5
    ├─ Stage 4: Background Novelty Gate (main process)
    │  └─ drop candidates matching empty workspace
    ├─ Stage 5: Session-Aware Tracking (BoT-SORT or ByteTrack)
    │  └─ 4 reset signals: frame gap, valley split, centroid jump, ReID shift
    ├─ Stage 6: Lazy GPU Refinement (Kornia perspective warp)
    │  └─ warp to 750×1050 portrait, optional 180° rotation
    ├─ Stage 7: Quality Scoring + Track Pruning
    │  └─ 7-component score: sharpness, glare, aspect, size, complexity, border_purity, confidence
    ├─ Stage 8: Front/Back Resolution (per session)
    │  └─ longest track = Front; pHash-gated Back (Hamming ≤ 22/64)
    ├─ Stage 9: Lighting-Diverse Selection + Median/Fusion
    │  └─ 4 quadrant glare centroids, select 3-4 sharpest, median-fuse
    │  └─ foil-aware: ECC register before detection, use glare-rejection for foils
    └─ Stage 10: Global Dedup + Storage (pHash inter-video, ReID cosine)
       └─ SQLite cards.sqlite, pipeline_events, telemetry

Key: Stages 1-3 run in producer/consumer subprocesses (decouples ffmpeg/GPU from orchestration).
     Stages 4-10 run in main process (thread-safe SQLite writes).
```

---

## 3. Critical Modules

### 3.1 **Sampler** (`sampler/__init__.py`)
- **Purpose:** Convert multi-minute video into sparse frame indices
- **Pass 1:** Fast 192px scan at 15fps → presence metrics (Sobel, empty-ratio, edge-density)
- **Pass 2:** Presence scoring (MobileNetV3-Small classifier OR Otsu fallback)
- **Pass 2b:** Valley-split detection (Sobel valleys + delta-spike clustering) → catches rapid card swaps
- **Output:** `FrameSample(frame_index, timestamp_ms, image, w, h)` for high-res decode + inference
- **Key Knobs:** `fast_scan_fps=15.0`, `presence_threshold=0.5`, `valley_drop_ratio=0.40`, `valley_min_width_frames=3`

### 3.2 **Detectors** (`detectors.py`)
- **Model:** YOLOv8-OBB (Oriented Bounding Box) from `AlecKarfonta/cardcaptor-v3`
- **Inference:** Resize to 640px longest-edge, batch size 16, conf threshold 0.5
- **Output:** 4-corner polygons + confidence + metadata
- **Critique:** Single global confidence threshold → misses partial/corner cards; no projective consistency check; no multi-instance NMS beyond YOLO's internal

### 3.3 **Background Novelty Gate** (`presence/background_novelty.py`)
- **Purpose:** Drop candidates that match empty workspace
- **Model:** Single mean grayscale from "empty" frames
- **Gate:** `|diff| / 255 > 0.08` (20 gray levels) per quad interior
- **Gating Points:** Candidate-level (Stage 4) + Track-level post-refinement (Stage 7)
- **Critique:** Single mean has no variance model; lighting drift softens gate; static-prop false positives

### 3.4 **Tracking** (`tracking/botsort_adapter.py`, `tracking/bytetrack_adapter.py`)
- **Backend:** BoT-SORT (default, with OSNet-x0.25 ReID) or ByteTrack (fallback, no ReID)
- **Session Reset Signals:**
  1. **Sampler frame-index gap** (adaptive per video via inter-window gaps)
  2. **Valley split** (hand-swap detected in Stage 1)
  3. **Centroid jump** (0.30× frame width) on highest-scoring candidate
  4. **BoT-SORT ReID shift** (identity change detected by appearance backbone)
- **Track Lifecycle:** UUID-4 `instance_id` per first sight, store candidates, compute angle (Front/Back)
- **Critique:** BoT-SORT fed dummy image for ReID → embeddings degraded; axis-aligned envelope discards orientation; centroid on bbox not card center

### 3.5 **GPU Refinement** (`gpu_refinement.py` → Kornia; `cropper.py` → CPU fallback)
- **Input:** 4 corners in source frame, full-res frame
- **Process:** Order corners clockwise → orient for portrait canvas → `getPerspectiveTransform` → `kornia.warp_perspective`
- **Output:** 750×1050 BGR uint8, optionally 180° rotated
- **Critique:** Fixed 750×1050 + GPU bilinear loses sub-pixel content; no Lanczos on GPU; no auto up-orientation

### 3.6 **Quality Scorer** (`scoring.py`)
- **Components:** `sharpness (25%), glare (15%), aspect_ratio (15%), size (10%), complexity (10%), border_purity (20%), confidence (5%)`
- **Border Purity (new v4.1):** Low ring variance ⇒ clean card border; high ⇒ finger intrusion
- **Per-Track Pruning:** Drop tracks where median quad-novelty < 0.08 (matches empty workspace)
- **Critique:** Glare ignores spatial structure; aspect tolerance ±25% too loose; all weights hand-tuned; size term saturates; no learned ranker

### 3.7 **Foil Detection & Fusion** (`fusion/foil_detection.py`, `fuser.py`)
- **Foil Detection:** Laplacian variance across frames (holographic shimmer = high freq energy shift)
  - Default threshold: 50.0 (calibrated on synthetic fixtures)
  - **Critical:** Detection happens BEFORE ECC registration (ECC bilinear low-passes the signal)
- **Fusion Strategy:**
  - **Non-foil:** Median fusion (per-pixel median across selected frames)
  - **Foil:** Glare-rejection fusion (adaptive saturation-aware blend)
  - **Selection:** 4 quadrants by glare centroid → pick sharpest per quadrant
- **Recent Work (Wave 4):** Added foil-aware luminance-distance glare rejection; tuned threshold via labeled fixtures

### 3.8 **Deduplicator** (`deduplicator.py`)
- **Perceptual Hash (pHash):** Inner 60% → 32×32 resize → DCT → 8×8 threshold-median → 64-bit hash
  - Hamming distance
  - Within-session threshold: ≤ 6/64 (≈9%)
  - Front/Back gate: ≤ 22/64 (≈34%)
- **ReID Embedding (BoT-SORT only):** Cosine distance on OSNet output, threshold 0.15
- **Critique:** pHash on inner 60% discards edge discriminators (logos, photos); threshold-median has exactly 32 bits set; ReID on degraded embeddings

### 3.9 **Storage** (`storage.py`)
- **Schema:** Cards.sqlite with `CardInstance`, `View`, `pipeline_events`, source evidence
- **Telemetry:** `run_telemetry.json` (timings, counts), `tracker_association_events.json` (tracking actions)
- **Output Structure:**
  ```
  <output_dir>/
    frames/           source frames (one per detection)
    crops/            canonical + fused views
    cards.sqlite      all metadata + events
    run_telemetry.json
    tracker_association_events.json
  ```

---

## 4. Configuration & Tuning Knobs

**Core Sampler:**
- `fast_scan_fps = 15.0` — scan speed vs detail trade-off
- `presence_threshold = 0.5` — MobileNetV3 classifier gate
- `valley_drop_ratio = 0.40` — Sobel valley sensitivity
- `delta_spike_ratio = 0.50` — delta-score swap detection

**Detector & Geometry:**
- `corner_confidence = 0.5` — YOLO confidence gate (global, fixed)
- `detection_width = 640` — inference size
- `empty_pixel_threshold = 0.98` — triage gate (Stage 2)

**Tracking:**
- `centroid_jump_ratio = 0.30` — jump threshold (0.30× width)
- `centroid_jump_frames = 3` — recent deque length
- `reid_distance_threshold = 0.6` — ReID embedding gate
- `min_track_length = 12` — adaptive ceiling (computed per-video)
- `tracker_backend = "botsort"` — BoT-SORT or ByteTrack

**Refinement & Scoring:**
- `rotate_180 = False` — camera mounted upside-down flag
- Quality score weights (see §3.6)
- `_CANONICAL_TARGET_FRAMES = 3` — target selected frames per track
- `_CANONICAL_MAX_FRAMES = 4` — max selected frames

**Fusion:**
- `foil_threshold` — Laplacian variance threshold (None = always median, 50.0 = foil-aware)

**Dedup:**
- `_SAME_CARD_HAMMING_MAX = 22` — Front/Back pHash gate
- `_SESSION_DUPLICATE_HAMMING_MAX = 6` — within-session pHash gate
- `background_novelty_threshold = 0.08` — quad-novelty gate (both Stages 4 & 7)

**Adaptive Per-Video:**
- `tracker_t_high` = 65th percentile of candidate confidence, clipped [0.40, 0.75]
- `tracker_t_low` = max(0.20, tracker_t_high - 0.20)
- `adaptive_min_track_length` = max(3, median_inter_gap_frames × 3)
- `adaptive_session_gap` = clip(p95 + safety_pad, floor=0.5s, cap=3.0s)

---

## 5. Data Flow & Key Types

```python
# Stage 1 Output
FrameSample(frame_index: int, timestamp_ms: int, image: np.ndarray, w: int, h: int)

# Stage 2 Output (after triage)
FramePacket(triage_metrics: dict, telemetry: dict)

# Stage 3 Output
DetectionPacket(
    frame_index, timestamp_ms, w, h,
    corner_detection: CornerDetection(corners: 4×Point, confidence: float),
    telemetry: dict
)

# Stage 4+ Processing
ScoredCandidate(
    detection_id, timestamp_ms, image_path, score: QualityScore,
    corners: 4×Point, frame_index: int
)

# Tracker Output
TrackState(
    instance_id: UUID, candidates: List[ScoredCandidate],
    last_centroid: Tuple, last_frame_index: int,
    missed_frames: int, active: bool, angle: str, reid_embedding: np.ndarray
)

# Session Resolution Output
_PreparedTrack(
    track: TrackState, session_id: int, angle: str,
    canonical_entries: List[np.ndarray], primary_hash: str,
    side_score: float, appearance_vector: np.ndarray
)

# Quality Score Components
QualityScore(
    sharpness, glare, aspect_ratio, size, complexity,
    border_purity, confidence, total
)
```

---

## 6. Recent Work (Wave 3/4)

### Completed
1. **Foil Card Detection** — Laplacian variance across frames to detect holographic shimmer
2. **Glare-Rejection Fusion** — Adaptive saturation-aware blending for foil cards
3. **Luminance-Distance Rejection** — C3 (luminance-distance calibration) for robust glare filtering
4. **Foil Threshold Calibration** — Labeled fixture sets (C2, E3) tuned threshold to 50.0
5. **Hard-Case Capture** — Active learning pipeline for edge cases (blur, occlusion, glare)
6. **Per-Video Adaptive Thresholds** — Tracker gates, session-split timing, track-length baselines computed from video stats
7. **Border Purity Scoring** — Detects finger intrusions in rectified crops

### Pending/Critique Points
1. **Background Model Variance** — Single mean insufficient; per-pixel variance + running average for lighting drift
2. **pHash Front/Back Gate Wrong Invariant** — Uses rotation tolerance (22/64 bits) to work around contour-only similarity; should use content-based metric
3. **Detector Confidence Global** — Single threshold misses partial/corner cards; per-region confidence or ROI-aware detection needed
4. **BoT-SORT ReID Degraded** — Fed dummy image; real crops would improve embedding quality
5. **Quality Scorer Hand-Weighted** — No learned ranker; `size` term saturates immediately
6. **Sampler Classifier Frame-Level** — No detection-conditioned localization signal; bystander cards in corners activate same as centered
7. **Multi-Process Error Handling** — Producer crashes silently on MPS/CUDA initialization; no structured error codes
8. **Fixed 750×1050 Canvas** — Bilinear interpolation for grading-downstream tasks; Lanczos+higher-res defensible

---

## 7. Testing & Evaluation

### Unit Tests
```bash
pytest tests/
```
Key test files:
- `test_pipeline.py` — full pipeline integration
- `test_deduplicator.py` — pHash logic
- `test_cropper.py` — perspective warp
- `test_presence_classifier.py` — MobileNetV3 inference
- `test_wave3_calibration.py` — adaptive threshold computation
- `test_centroid_jump.py` — tracking reset signal

### Telemetry
- `run_telemetry.json` — per-stage timings, frame counts, detection ratios
- `tracker_association_events.json` — every new_track/assigned_existing action
- `cards.sqlite` — `pipeline_events` table with session reset reasons

### Offline Inspection
```bash
card-capture sampler sessions <video>  # Stage 1 only, ~35s vs 2+ min full pipeline
card-capture harness run                # regression report vs baseline JSON
```

---

## 8. Command Reference

### Process a Video
```bash
card-capture process <video> \
  --output-dir <dir> \
  --db <db.sqlite> \
  --detector docaligner \
  --reader-backend auto \
  --queue-size 256 \
  --inference-batch-size 16 \
  --corner-confidence 0.5 \
  --device auto
```

### Review UI
```bash
card-capture review --db <db.sqlite> --port 8000
```
Then open `http://localhost:8000`.

### Smoke Test (Fake Detector)
```bash
card-capture process <video> \
  --output-dir <temp-out> \
  --db <temp-db> \
  --detector fake \
  --reader-backend auto
```

---

## 9. Common Next Steps

### Feature Development
- Improve pHash front/back detection (rotation + content-aware metric)
- Add per-region detector confidence (handle corner/partial cards)
- Real ReID embeddings for BoT-SORT (pass unregistered frames)
- Learned quality ranker (replace hand-weighted sum)
- Per-pixel background model with variance

### Debug & Calibration
- Add verbose telemetry on rejection reasons (why was candidate dropped?)
- Surface producer subprocess errors (MPS/CUDA init failures)
- Adaptive foil threshold per video class (glossy vs holographic)
- Tune border-purity weights on reviewer feedback

### Performance
- GPU batch fusion (stack frames on device before median)
- Higher-res canvas (1000×1400) with Lanczos on GPU
- Parallel Stage 9 (per-track fusion in thread pool)
- Incremental background model (online mean/variance)

---

## 10. Architecture Critique Summary

| Issue | Impact | Possible Fix |
|-------|--------|--------------|
| Single detector confidence threshold | Misses partial/occluded cards | Per-region confidence, ROI-aware detection |
| pHash Front/Back gate uses wrong invariant | 34% bit tolerance to work around | Content-based metric (texture diff) |
| BoT-SORT ReID on dummy image | Embeddings degraded | Pass real frames to appearance backbone |
| Background model = single mean | Lighting drift softens gate | Per-pixel Gaussian + running average |
| Quality score hand-weighted | Size term saturates, glare ignores spatial structure | Learned ranker on reviewer labels |
| Sampler classifier frame-level | Bystander cards in corners activate equally | Detection-conditioned ROI or objectness map |
| Multi-process error handling | Producer crashes silently | Structured error codes + retry backoff |
| Fixed 750×1050 + GPU bilinear | Sub-pixel content lost for grading tasks | Higher-res canvas (1000×1400) + Lanczos |

---

## 11. Files to Know

```
src/card_capture/
├─ pipeline.py                      Main orchestration, stages 4-10
├─ sampler/__init__.py              Stage 1, presence + valley splits
├─ sampler/valley_splits.py         Valley detection algorithm
├─ ingestion.py                     Stage 2, frame triage
├─ detectors.py                     Stage 3, YOLO inference
├─ presence/
│  ├─ classifier.py                 MobileNetV3-Small presence detector
│  └─ background_novelty.py         Stage 4, novelty gate
├─ tracking/
│  ├─ botsort_adapter.py            BoT-SORT backend + ReID
│  ├─ bytetrack_adapter.py          ByteTrack backend
│  └─ centroid_jump.py              Reset signal #3
├─ gpu_refinement.py                Stage 6, Kornia GPU path
├─ cropper.py                       Stage 6, CPU fallback
├─ scoring.py                       Stage 7, quality components
├─ fuser.py                         Stage 9, selection + fusion
├─ fusion/
│  ├─ foil_detection.py             Laplacian variance foil detection
│  └─ median_fusion.py              Glare-rejection fusion
├─ deduplicator.py                  Stage 10, pHash + ReID
├─ storage.py                       SQLite schema + persistence
├─ config.py                        Config dataclass + loading
└─ adaptive_gap.py                  Session-split gap computation

tests/
├─ test_pipeline.py
├─ test_deduplicator.py
├─ test_cropper.py
├─ test_presence_classifier.py
├─ test_wave3_calibration.py
├─ test_centroid_jump.py
└─ ... (others)

models/
├─ presence_classifier.pt           MobileNetV3-Small weights (optional)
└─ cardcaptor_v3_best.pt            YOLOv8-OBB weights (HuggingFace)

scripts/
├─ calibrate_foil_threshold.py      Tune foil detection threshold
├─ generate_foil_fixtures.py        Create labeled foil/non-foil sets
├─ calibrate_wave3.py               Adaptive threshold sweep
└─ ...

Docs:
├─ PIPELINE_V3_OVERVIEW.md          High-level v3 flow
├─ arch-4.1.md                      Exhaustive v4.1 specification
├─ v3.md, v2.3.md, v2.2.md          Historical versions
└─ QUICK_REFERENCE.md               CLI flags and quick start
```

---

## 12. How to Use This Document

This is your **context reload** for future sessions. When starting work:
1. Read **Section 1-2** for problem + architecture
2. Skim **Section 3** for module map
3. Reference **Section 4** for tuning knobs if tweaking thresholds
4. Check **Section 6** for recent work + pending critique
5. Jump to **Section 11** for file locations when diving into code
6. Use **Section 9** to prioritize next work

When you have clarified an upgrade path, **append it below** as a new section.

---

## Appendix A: Upgrade Path — v4 Surgical Fixes + Application Shell

This appendix records the chosen direction after extensive review. It is the
authoritative roadmap; supersedes any conflicting earlier note.

### A.0 Strategic Position

**Decision: preserve the algorithm library, refactor the orchestration layer,
build a real application shell on top.** Not a ground-up rewrite, not a
CLI-only patch.

The empirical justification:

| Layer | Verdict | Why |
|---|---|---|
| Algorithm modules (sampler, detectors, scoring, fuser, foil_detection, ECC, deduplicator, presence/, tracking/, gpu_utils, cropper) — 50-470 lines each | **Preserve** | Well-decomposed, encodes hard-won corner-case knowledge (foil detection, valley splits, ECC registration, adaptive thresholds, border purity). Replacing it gains nothing concrete. |
| `pipeline.py` (2,079 lines, single file) | **Decompose** | Monolithic orchestration. The "clean architecture" is a façade — the control flow is tangled in one giant file. Refactoring into named stages with explicit interfaces is justified independently of any algorithm change. |
| `storage.py` (574 lines, SQLite) | **Preserve + extend** | Data model is sound (videos, card_instances, card_views, pipeline_events). Wrap in a service layer. |
| `review.py` + 4 HTML templates | **Replace UI; keep FastAPI** | Stack choice is right; the table-and-dropdown UX is not. |
| `cli.py` (527 lines) | **Keep alongside UI** | CLI stays for headless/CI; app sits on the same service layer. |

This neutralizes the second-system risk (algorithms preserved, A/B baseline
intact) while addressing what's actually broken (orchestration tangle, basic
UX, missing training infrastructure).

### A.1 Phase Plan

Each phase is independently shippable and verifiable against the regression
harness from Phase 0.

#### Phase 0 — Regression Harness (BLOCKER for everything else)

Nothing in Phases 1+ ships without this. It's the single highest-leverage
piece of work and it's framework-independent.

Deliverables:
- Truth-file schema codified (`truth.json` per video): `expected_cards[]`
  with `card_id`, `front_present`, `back_present`, `approx_front_window_ms`,
  `approx_back_window_ms`, `physical_card_key`, `is_foil`, `notes`.
- Metric definitions:
  - **Card recall:** detected cards / ground-truth cards
  - **Card precision:** real detections / total detections (phantom rate)
  - **Side accuracy:** correct front/back assignments / total instances
  - **Dedup accuracy:** correctly grouped duplicates / ground-truth groups
    (Adjusted Rand Index or F1 on pairs)
  - **Image quality:** SSIM / PSNR of fused canonical vs. a hand-picked
    reference frame
- `card-capture harness run --against <baseline.json>` CLI subcommand.
- Web UI: Regression tab (Section A.4.6) shows per-metric deltas with
  per-video breakdown and highlight on regressions.
- Initial labeled set: **15 videos minimum**, covering: clean run, glare,
  foil, hand occlusion, fast swaps, edge-on flips, dark workspace, bright
  workspace, mixed orientations, partial visibility, multi-card-in-frame.

Acceptance: harness produces stable metrics across 3 consecutive runs on the
same video (no noise from non-determinism in pipeline).

#### Phase 1 — Application Shell + Labeling UX

Build the operator app and the labeling surfaces in parallel with Phase 0,
because (a) Phase 0 needs the labeling UX to be efficient, and (b) the app
shell unblocks all subsequent work.

Deliverables:
- FastAPI service layer wrapping `Storage` and `VideoProcessor`
- SSE/WebSocket endpoint for real-time pipeline progress
- Frontend: pick **Svelte or HTMX + Alpine.js** (low-ceremony, no React
  build-pipeline). Single-page app, left-nav sections (A.4)
- Labeling UX (A.3) — filmstrip, three-button verdict, hotkeys, drag-link,
  Front/Back trainer, dedup-cluster confirmer
- Real-time run monitor with per-stage progress

Acceptance: a human can label a 5-minute video's truth file in < 10 minutes
using only mouse + keyboard, no JSON editing.

#### Phase 2 — pipeline.py Decomposition

Break the 2,079-line monolith into named stage modules with explicit
interfaces. This is pure refactor — behavior preserved, gated by Phase 0.

**Orchestration choice: custom Stage protocol (~250 LOC), not a library.**

Rationale: Our pipeline is (a) single-user, single-machine, linear DAG with
no parallelism, (b) stream subsystem (Stages 1–3) + batch DAG (Stages 4–10),
and (c) needs per-stage artifact persistence for the threshold-tuning
playground. A lightweight Stage protocol with in-process artifact store
matches this shape exactly; heavyweight orchestrators (Prefect, Airflow,
Metaflow) add ceremony with no payoff. Metaflow is documented as the exit
ramp if future work requires distributed compute or richer artifact lineage.

Deliverables:
- `pipeline/stage.py` — Stage protocol (input → output), StageContext
  (run_id, config, artifacts store, telemetry), Pipeline class (run with
  optional resume_from parameter)
- `pipeline/orchestrator.py` — < 100 lines, only stage sequencing + timing
- `pipeline/artifact_store.py` — persist stage outputs to `<run_dir>/artifacts/<stage_name>`
- `pipeline/stages/` directory with one file per stage (s1_streaming,
  s2_triage, s3_detector, ... s10_dedup_storage). Stages 1–3 wrapped as
  single "streaming" stage with internal multiprocessing; Stages 4–10 are
  individual stages
- All persisted artifacts unchanged; harness must report 0% delta on golden set

Acceptance: harness metrics identical of pre-refactor baseline; orchestrator
code < 100 lines; Stage protocol self-documents the pipeline shape; resume
from any stage works and skips re-runs.

#### Phase 3 — High-Impact Algorithmic Fixes

These are the expert's prioritized list. Each ships independently and is
gated by Phase 0 regression metrics.

Order is by expected accuracy delta × implementation cost:

1. **Multi-frame median fusion** (already implemented in v3 `fuser.py`; verify
   it's enabled, tune frame-count, possibly add residual-region inpainting).
   *Expected: noticeable glare reduction, especially on non-foil cards.*
2. **Trained Front/Back classifier** replacing
   `longest-track-equals-Front` heuristic.
   - Model: MobileNetV3-Small finetuned on rectified crops
   - Training data: collected via Labeling UX (A.3.2)
   - Threshold: confidence < 0.6 → fall back to current heuristic
   *Expected: biggest win on side correctness, your #1 reported pain point.*
3. **DINOv2 embeddings + local FAISS for dedup** replacing pHash.
   - Model: DINOv2 ViT-S/14 (~22M params), CoreML if on Apple, else PyTorch
   - Index: FAISS in-process, no Qdrant
   - Threshold: cosine distance, calibrate on labeled dedup groups
   *Expected: large reduction in both false-positive dedup ("two parallels
   collapsed") and false-negative dedup ("same card not matched").*
4. **ByteTrack** (or fixed BoT-SORT) replacing current adapter:
   - Decision point: BoT-SORT with real-image ReID, or ByteTrack with no ReID
     (avoid the dummy-image-degraded ReID problem entirely)
   - Removes hand-tuned shape-change rules
   *Expected: cleaner tracks, fewer ID switches, less session fragmentation.*
5. **Corner refinement (RANSAC line-fit) on canonical frames** — sub-pixel
   accurate corners before rectification.
   *Expected: sharper rectified crops; marginal but compounds with fusion.*

#### Phase 4 — Speed Wins (no accuracy change expected)

Each is independently shippable; not gated by accuracy regression but by
"no degradation."

1. YOLOv8-OBB → **YOLO26-OBB on CoreML** (Apple silicon target; PyTorch
   fallback for Linux/CUDA preserves cross-platform)
2. Decoder: OpenCV/decord → **VideoToolbox on macOS** (preserve current
   backends elsewhere)
3. Perspective warp: Kornia → **vImage on macOS** (Kornia fallback elsewhere)

Cross-platform is preserved by feature-detecting macOS-only paths at startup,
not by replacing the cross-platform code.

#### Phase 5 — Hard-Case Active Learning Loop

The plumbing exists (`analysis/hard_case_capture.py`). Wire it into the app:
- Hard cases captured during runs surface in the **Hard Cases** tab
- Operator one-click "send to training set" → retrains relevant model
- Track regression metric per model retrain

### A.2 What's Explicitly Skipped (and why)

| Skipped | Why |
|---|---|
| `VNDetectRectanglesRequest` (Vision Framework) | Classical CV with known failure modes on low-contrast edges, glare-crossed borders, hand occlusion, foil refraction, severe perspective. YOLO already handles these. |
| Single-frame neural glare removal (HDNet/ACENet/DRM solvers) | Ill-posed on non-Lambertian surfaces. Holographic cards have specular patterns that ARE the texture — DRM solvers would hallucinate. Multi-frame median is principled and free. |
| Prefect / Airflow orchestration | Enterprise data-engineering for a single-user batch pipeline. `multiprocessing` + service layer is sufficient. |
| OpenTelemetry / Prometheus / Grafana | Same — solving a problem that doesn't exist yet. `py-spy` and SQLite telemetry are enough. |
| Qdrant | FAISS in-process gives sub-ms search for ≤ 100K cards with zero ops burden. |
| MLX for inference | CoreML on ANE is faster for forward passes. Use MLX only if training is added later. |

### A.3 Labeling UX Spec

#### A.3.1 Per-video truth labeling (replaces `templates/labeling.html`)

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ Video: practice_session_03.mov   [< prev]  [next >]  [save] │
├─────────────────────────────────────────────────────────────┤
│  [video scrubber with detected-card markers]                │
├─────────────────────────────────────────────────────────────┤
│  Detected instances (12)             [✓ all real]  [✗ all]  │
│  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐         │
│  │F1│ │B1│ │F2│ │B2│ │F3│ │ ? │ │F4│ │F5│ │B5│ │F6│ ←sel    │
│  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘         │
│   ✓    ✓    ✓    ✓    ✓    ✗?    ✓    ✓    ✓               │
├─────────────────────────────────────────────────────────────┤
│  Selected: instance #6                                       │
│  ┌─────────────────────┐  [✓ Real]  [✗ Phantom]  [🔄 Flip]  │
│  │   <large preview>   │  Pipeline auto-assigned: Front      │
│  │  fused canonical    │  Hotkeys: F=Front  B=Back  X=phantom│
│  └─────────────────────┘  Linked to: (none — drag to link)  │
├─────────────────────────────────────────────────────────────┤
│  Missed cards: [+ Add at current scrubber time]              │
│  • card_missed_1  at 04:23-04:31 [edit][delete]              │
└─────────────────────────────────────────────────────────────┘
```

**Interactions:**
- Click thumbnail → select instance, large preview loads
- F/B/X keys → set verdict and auto-advance to next unverified instance
- Drag instance onto another → mark as same card (visual: shared color border)
- Right-click → context menu (split group, mark canonical-for-group, view raw frames)
- Scrubber + "Add missed card" button → fills `card_missed_N` with current
  timestamp, opens inline form for end-timestamp and side

**Save:** auto-save every 30s + explicit save button; produces same
`truth.json` schema today's labeling.html produces (backward compatible).

#### A.3.2 Front/Back classifier training UI

**Single-card flash-card mode:**
```
┌─────────────────────────────────────────────────────────────┐
│ Train Front/Back classifier      Labeled: 247 / target 500  │
│                                                              │
│         ┌───────────────────────────────┐                   │
│         │                               │                   │
│         │     <single rectified card>   │                   │
│         │     (750×1050, full size)     │                   │
│         │                               │                   │
│         └───────────────────────────────┘                   │
│                                                              │
│  [F = Front]  [B = Back]  [S = Skip ambiguous]  [U = Undo]  │
│                                                              │
│  Source: practice_session_03.mov / instance_42 / frame_1287 │
│  Auto-retrain after 50 more labels                          │
└─────────────────────────────────────────────────────────────┘
```

**Interactions:**
- Single keypress (F/B/S) labels and advances. No mouse needed.
- Cards drawn from un-labeled high-confidence detections across all videos
- "U" undoes last label
- Auto-retrain triggers at configurable interval (default 50 new labels)
- Validation accuracy reported after each retrain

#### A.3.3 Dedup-cluster confirmation

**Cluster grid mode:**
```
┌─────────────────────────────────────────────────────────────┐
│ Verify dedup clusters    23 clusters, 7 unverified          │
│                                                              │
│ Cluster #5 (predicted: 3 instances of same card)            │
│ ┌──┐ ┌──┐ ┌──┐                                              │
│ │  │ │  │ │  │   [✓ All same]  [Split selected to new]      │
│ └──┘ └──┘ └──┘   [+ Add card from inventory]                │
│  v3   v7   v12                                              │
│                                                              │
│ Cluster #6 (predicted: 2 instances) ← unverified            │
│ ...                                                          │
└─────────────────────────────────────────────────────────────┘
```

**Interactions:**
- Click thumbnail to select; multi-select with Shift+click
- "✓ All same" — confirms cluster, moves to next
- "Split selected to new" — pulls selected cards into a new cluster
- Drag-drop a card onto another cluster → merge

### A.4 Application Interface Spec

Left-nav sections, each a single-page route:

#### A.4.1 Inbox
- Drag-drop video upload (multi-file)
- Queue: pending, processing, completed, failed
- One-click "Run pipeline" with config preset selector (fast / balanced / quality)
- Recent runs summary card with progress

#### A.4.2 Runs
- List of all runs (video × config × timestamp)
- Per-run status, cards extracted, elapsed time, comparison-to-baseline badge
- Click → run detail page (timeline, telemetry, cards, rejection log)
- "Compare to..." button → A/B view (A.5.2)

#### A.4.3 Cards
- Grid view of all extracted cards across all runs
- Filters: run, video, dedup-group, review-state (pending/accepted/rejected),
  side (front/back), is_foil, confidence range
- Bulk actions: accept-all-filtered, reject-all-filtered, export
- Click card → detail (canonical view, fused view, source frames,
  detection metadata, dedup neighbors)

#### A.4.4 Label
- Sub-tabs: Per-video truth (A.3.1), Front/Back trainer (A.3.2),
  Dedup clusters (A.3.3)
- Top-level progress: "15/30 videos labeled, 247/500 F/B examples,
  23/30 dedup clusters verified"

#### A.4.5 Train
- Dataset statistics per model (presence, F/B, future: dedup-similarity)
- Model status: last trained, validation accuracy, version
- "Retrain now" button + auto-retrain settings
- Validation set previews (model wrong here ← click to inspect)

#### A.4.6 Regression
- Pick a candidate run config and a baseline (default: tagged `baseline_v3`)
- Run on full golden set, get table of per-video metric deltas
- Per-metric drill-down (recall, precision, side accuracy, dedup accuracy)
- Highlight regressions in red; gate "Promote to baseline" button on
  no-regression

#### A.4.7 Settings
- Pipeline config presets (fast / balanced / quality / custom)
- Threshold sliders with tooltips explaining trade-offs
- Live preview: pick a saved run, drag a threshold → see recomputed
  rejection/acceptance (uses persisted stage outputs from Phase 2)

### A.5 Dev/Operator Dashboard Spec

#### A.5.1 Run inspection (click any run from A.4.2)

Tabbed view:
- **Timeline:** v3's existing timeline.html refreshed — sessions, resets
  (color-coded by reason), instances, valley splits
- **Cards:** all cards from this run with side-by-side canonical/fused
- **Telemetry:** per-stage timing chart, throughput, frame counts,
  rejection breakdown by stage
- **Events log:** every pipeline_event row with filterable reasons
  (sampled_frame_gap, valley_split, centroid_jump, reid_shift,
  novelty_below_threshold, low_confidence)
- **Rejection log:** every candidate dropped, with stage + reason +
  thumbnail of the rejected crop. Click to "open in label UI as missed card"
  if it looks like a false rejection
- **Hard cases:** auto-captured edge cases from this run; one-click
  "send to training set"

#### A.5.2 A/B comparison view

- Pick run A and run B (same video, different config or pipeline version)
- Side-by-side: cards extracted, dedup groups, side assignments
- Diff highlighting: cards in A but not B (and vice versa),
  re-assignments, dedup-group changes
- Metric delta strip at top (recall ±X%, precision ±X%, etc.)

#### A.5.3 Threshold-tuning playground

Useful for: `corner_confidence`, `background_novelty_threshold`,
`_SAME_CARD_HAMMING_MAX`, `foil_threshold`, `centroid_jump_ratio`,
`valley_drop_ratio`.

Mechanics:
- Pick a saved run from A.4.2
- Drag a threshold slider → backend recomputes downstream stages from
  persisted intermediate outputs (not re-running detector/sampler)
- Live updated metrics + thumbnail strip of newly accepted/rejected
- "Commit as new config preset" button

Requires Phase 2's stage decomposition (persisted per-stage outputs).

### A.6 Service Layer Architecture

```
┌──────────────────────────────────────────────────────┐
│  Frontend (Svelte or HTMX+Alpine, served by FastAPI) │
└──────────────────────┬───────────────────────────────┘
                       │ REST + SSE
┌──────────────────────▼───────────────────────────────┐
│  FastAPI service layer (app/api/)                    │
│  - /api/videos   /api/runs   /api/cards              │
│  - /api/label    /api/training  /api/regression      │
│  - /events/<run_id>  (SSE: per-stage progress)       │
└──────────────────────┬───────────────────────────────┘
                       │ in-process calls
┌──────────────────────▼───────────────────────────────┐
│  Domain services (app/services/)                     │
│  - PipelineService (wraps VideoProcessor)            │
│  - LabelingService (truth.json CRUD)                 │
│  - TrainingService (model retrain, validation)       │
│  - RegressionService (harness, metrics, baselines)   │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  Existing pipeline + storage (preserved)             │
│  - src/card_capture/pipeline/* (post-Phase 2 stages) │
│  - src/card_capture/storage.py (extended schema)     │
│  - src/card_capture/<algorithm modules>              │
└──────────────────────────────────────────────────────┘
```

CLI (`cli.py`) calls the same domain services. Headless / CI workflows
continue to work.

### A.7 Implementation Order (concrete)

The next 6-8 weeks of work, gated by Phase 0:

1. **Week 1-2:** Phase 0 — truth-file schema, metric definitions, harness
   CLI command, minimal `Regression` tab. Label 5 videos to bootstrap.
2. **Week 2-3:** Phase 1 — FastAPI service layer, SSE progress, app shell
   (left-nav, Inbox, Runs sections), per-video labeling UX (A.3.1).
3. **Week 3-4:** Phase 1 cont. — F/B trainer UX (A.3.2), dedup cluster UX
   (A.3.3), Cards section, Train section. Reach 15 labeled videos.
4. **Week 4-5:** Phase 2 — pipeline.py decomposition, verified by harness
   showing 0% delta.
5. **Week 5-6:** Phase 3 #1-2 — multi-frame fusion verification + Front/Back
   classifier (data already collected during Phase 1).
6. **Week 6-7:** Phase 3 #3 — DINOv2 + FAISS dedup. Regression run, gate.
7. **Week 7-8:** Phase 3 #4-5 — tracker swap, corner refinement.
   Phase 4 speed wins as time allows.

Phase 5 is ongoing once Phase 1 ships.

### A.8 Orchestration Library Decision

**Chosen: custom Stage protocol (~250 LOC in-tree), not a framework.**

Considered: Metaflow (strong library candidate; artifact persistence + resume
are native), Hamilton (lighter but less stream-friendly), Kedro (too
ceremonious), Luigi/Snakemake (file-driven, awkward for frames), Ray/Dask
(distributed overkill), Prefect/Airflow (explicitly ruled out as too heavy).

**Why custom Stage protocol wins for this problem:**
1. Pipeline shape (linear DAG, no parallelism, stream subsystem + batch DAG)
   is specific enough that a lightweight protocol matches exactly.
2. Artifact persistence is load-bearing for the threshold-tuning playground
   (§A.5.3); we need it persisted per-stage anyway, so no external dependency.
3. Resume capability (skip re-runs from a given stage) is ~30 extra lines in
   the Pipeline class.
4. Zero framework lock-in; we control the entire orchestration shape.

**Exit ramp:** If future work requires distributed compute or richer
artifact lineage (multi-branch DAGs, fan-out/fan-in), the Stage protocol can
be adapted to `@step` decorators in Metaflow with minimal migration. Document
this as the intended evolution path in Phase 2.

### A.9 Open Questions for Future Sessions

These were flagged but not yet decided:

1. **Frontend framework:** Svelte vs HTMX+Alpine? Svelte gives a richer
   labeling UX (drag-drop, real-time updates) but adds a build step.
   HTMX+Alpine keeps the FastAPI Jinja-template flow and is enough for
   tables + forms but constrained for the filmstrip/drag UX.
2. **DINOv2 variant:** ViT-S/14 (recommended for speed, ~22M params) vs
   ViT-B/14 (better accuracy, ~86M params). Decide after benchmarking on
   labeled dedup groups.
3. **Tracker choice:** BoT-SORT-with-real-ReID (more work, better
   discrimination on visual identity) vs ByteTrack-no-ReID (simpler,
   relies on spatial + appearance-via-Front-Back).
4. **Training infra:** local-only retrain on Apple silicon? Cloud GPU
   for retrain? Decide once dataset sizes are known.
5. **Apple-specific path:** YOLO26-CoreML, VideoToolbox, vImage are
   macOS-only. Confirm we want a feature-detected fast-path vs a
   universal slow-path, or hold these until cross-platform consensus.


