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

## Appendix: Upgrade Path (To Be Filled In)

*Awaiting user input on desired improvements and exploration direction.*

