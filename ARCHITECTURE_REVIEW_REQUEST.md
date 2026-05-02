# Card Capture: Architecture Review Request

**Project:** Sports Trading Card Image Extraction from Video  
**Date:** May 2, 2026  
**Status:** Post-Implementation, Seeking Architectural Improvements  
**Codebase Size:** ~2,100 lines of Python (core pipeline)

---

## Executive Summary

Card Capture is a video processing pipeline that automatically extracts high-quality images of trading cards from hand-held lightbox videos. The system has moved from proof-of-concept to a functional MVP with GPU acceleration and multi-metric detection. This document outlines the current architecture and requests architectural review for improvements in **detection accuracy** and **processing performance**.

**Current State:**
- ✅ Two-pass frame selection with contrast-based and detection-guided sampling
- ✅ GPU acceleration for variance/sharpness/edge density computation
- ✅ Multi-metric detection (variance, motion, edge density)
- ✅ YOLO-based card detection with quality scoring
- ✅ SQLite-based storage and web-based review UI
- ⚠️ Detection accuracy ~70% (7 cards in 10-card test video)
- ⚠️ Processing time ~13 minutes for 60-second video (M2 Mac)

---

## Problem Statement

### User's Workflow

Users film trading cards in a controlled **lightbox environment**:

1. Place card under overhead camera
2. Hold briefly (1-3 seconds) while camera records
3. Remove card
4. Repeat for each card (typical: 7-15 cards per session)
5. Upload video file for processing
6. Review extracted stills and manually approve/reject

### Current Challenges

**Challenge 1: Incomplete Detection**
- System detects ~70% of cards shown in video
- Typically misses 2-4 cards per 10-card video
- Example: 7 distinct cards → 4-5 detections + duplicates/false positives
- Impact: Manual review requires adding missing cards back in

**Challenge 2: Processing Speed**
- 60-second video takes 12-13 minutes to process
- M2 Mac with GPU available but underutilized
- Frame seeking (O(n) per frame) is a bottleneck
- Impact: Users wait too long for results

**Challenge 3: Parameter Tuning**
- Contrast threshold varies by lighting/card type (needs manual tuning)
- No visibility into per-stage decisions (what triggered detection?)
- Hard to diagnose why specific cards are missed

---

## Current Architecture

### System Overview

```
Input Video
    ↓
[FRAME SAMPLER]  ← Two-pass: find "card present" windows, select sharp frames
    ↓
Frame Candidates (FrameSample objects)
    ↓
[CARD DETECTOR]  ← YOLO or Fake detector (for testing)
    ↓
Detections (CardDetection objects)
    ↓
[CROPPER]        ← Extract polygon region from frame
    ↓
Card Crops (CropResult objects)
    ↓
[QUALITY SCORER] ← Six component scoring: sharpness, glare, aspect, complexity
    ↓
Scored Cards (QualityScore objects)
    ↓
[SELECTOR]       ← Time-group candidates, pick top N by score
    ↓
Best Cards (10 saved by default)
    ↓
Database + Output Directory
```

### Key Components

#### 1. **Sampler** (745 lines, `sampler.py`)
**Role:** Frame selection strategy

**Available Samplers:**
- `VideoSampler` — Every Nth frame at target fps (baseline)
- `StabilityBasedSampler` — Find low-motion windows (motion-based)
- `DetectionGuidedSampler` — Find YOLO-high-confidence windows (detection-based)
- `ContrastBasedSampler` — Find high-variance windows (contrast-based) ← **Current focus**

**ContrastBasedSampler Algorithm:**

*Pass 1 (Presence Detection):*
- Scan video at low resolution (160px wide, 5 fps)
- Compute RGB color variance per frame
- Mark frames with variance > threshold as "card present"
- Group consecutive frames into presence windows
- Filter windows by min_presence_frames (default 3)

*Pass 2 (Sharpness Scoring):*
- For each presence window, read full-resolution frames
- Compute Laplacian variance (sharpness) for each frame
- Select top N sharpest frames per window
- Yield as FrameSample objects

**Performance Characteristics:**
- Pass 1: ~60 seconds for 60-second video (sequential frame reading)
- Pass 2: ~180-600 seconds for 60-second video (full-res seeking + Laplacian)
- Total: 240-770 seconds (4-13 minutes)

**Current Issues:**
- Frame seeking is O(1) but overhead; queuing 14 presence windows × 5 candidates = 70 frame reads
- No adaptive batching or caching of candidate frames
- Video I/O not optimized for random access patterns

#### 2. **GPU Utils** (392 lines, `gpu_utils.py`)
**Role:** GPU-accelerated metric computation

**Metrics Computed:**
- **Variance:** Color variance in RGB space (used in Pass 1 contrast detection)
- **Sharpness:** Laplacian variance (used in Pass 2 frame ranking)
- **Motion:** Frame-to-frame optical flow magnitude (experimental)
- **Edge Density:** Sobel edge detection ratio (experimental)

**GPU Optimization:**
- PyTorch-based CUDA/MPS computation
- Batch processing of multiple frames
- Device auto-detection (CUDA/MPS/CPU)

**Current Issues:**
- Motion detection enabled by default (causes false positives in camera movement)
- Edge density threshold (0.15) never triggers in practice
- Multi-metric OR-fusion too permissive (any metric = detection)
- No visibility into metric distributions per video

#### 3. **Detector** (139 lines, `detectors.py`)
**Role:** Card localization and confidence scoring

**Available Detectors:**
- `FakeCardDetector` — Mock detector for testing (returns random boxes)
- `CardcaptorUltralyticsDetector` — YOLO-based real detection

**YOLO Integration:**
- Model: `ultralytics/yolov8n` (nano variant for speed)
- Confidence threshold: default 0.25
- Output: Card bounding polygons + confidence scores

**Current Issues:**
- YOLO trained on general objects, not trading cards specifically
- Misses cards at angles or with reflections
- False positives on lightbox edges/shadows
- Requires ~1-2 seconds per frame for full-resolution inference
- No model fine-tuning on actual trading cards

#### 4. **Quality Scorer** (58 lines, `scoring.py`)
**Role:** Multi-dimensional card image quality assessment

**Six-Component Scoring:**
1. **Sharpness** (Laplacian variance) — Prefers crisp, in-focus images
2. **Glare/Overexposure** (brightness ratio) — Penalizes washed-out cards
3. **Aspect Ratio** — Penalizes extreme foreshortening/occlusion
4. **Complexity** (texture count) — Rewards detailed card surfaces
5. **Motion Blur** (edge coherence) — Penalizes blurred edges
6. **Confidence** (detector score) — YOLO confidence weighting

**Aggregation:** Weighted geometric mean of components

**Current Issues:**
- Weights are hardcoded (0.2 per component)
- No adaptive weighting by card type or lighting
- Glare detection simplistic (only checks brightness range)
- Complexity metric uses HSV histogram bins (may miss detail in monochrome cards)

#### 5. **Selector** (39 lines, `selector.py`)
**Role:** Deduplication and ranking of candidate cards

**Algorithm:**
- Group candidates by timestamp (1000ms windows)
- Pick best candidate per group (highest quality score)
- Sort globally by score
- Return top N (default 10)

**Current Issues:**
- Rigid time-windowing may group distinct cards if timing is close
- No content-based deduplication (can save duplicate image of same card)
- Fixed max_candidates doesn't adapt to video length

#### 6. **Pipeline** (148 lines, `pipeline.py`)
**Role:** Orchestrate sampler → detector → scorer → selector

**Flow:**
1. Initialize sampler, detector, storage
2. For each FrameSample from sampler:
   - Run detector on frame
   - For each detection:
     - Crop card region
     - Score crop quality
     - Add to candidates list
   - Handle early-stop (optional)
3. Run selector on candidates
4. Save top N to database and output directory

**Current Issues:**
- Detector runs on every candidate frame (expensive)
- No batching of detector inference
- No frame caching between sampler and detector

---

## Performance Bottleneck Analysis

### Timing Breakdown (60-second video, M2 Mac, GTX 1080)

```
Pass 1 (Variance Scan):           ~60 seconds
  - Video I/O:                    ~40s
  - Variance computation (GPU):   ~10s
  - Frame grouping:               ~10s

Pass 2 (Sharpness Scoring):      ~180-600 seconds (highly variable)
  - Frame seeking:                ~150-550s (7-14 seeks, ~40s per frame read)
  - Full-res I/O:                 ~20s
  - Laplacian computation (GPU):  ~5s
  
Detection/Scoring Loop:           ~30 seconds
  - YOLO inference (1-2s per frame): ~10-20s
  - Quality scoring (GPU):        ~5s
  - Database ops:                 ~5s
  
Selection & Output:               ~5 seconds
```

**Key Insight:** Frame seeking and I/O dominate (71% of time).

---

## Detection Accuracy Analysis

### Test Case: IMG_5596.MOV
- **Video Length:** 60 seconds @ 30fps
- **Expected:** 7 distinct card fronts/backs
- **Current Result:** 4-5 cards detected + 1-2 duplicates/false positives

### Why Cards Are Missed

1. **Sampler Missing Windows (rare)**
   - Contrast threshold doesn't trigger for low-variance cards
   - Solution: Adaptive threshold or secondary metrics

2. **Detector Misses Cards (common)**
   - YOLO not trained on trading cards
   - Missed at angles, with reflections, or overlaps
   - Solution: Fine-tuned YOLO or alternative detection

3. **Selector Deduplication (occasional)**
   - Time-window may group separate card instances
   - Solution: Content-based deduplication

### Known Parameter Issues

- **contrast_threshold:** 600-1000 too low for some videos; need 2000-3000
- **motion_enabled:** False positives from camera movement; should be opt-in
- **edge_density:** Threshold (0.15) never triggers; remove or retune

---

## Data Flow and State

### Object Models

```python
FrameSample(frame_index, timestamp_ms, image, width, height)
  ↓
CardDetection(frame_index, timestamp_ms, polygon, confidence, label, metadata)
  ↓
CropResult(image, width, height, polygon)
  ↓
QualityScore(total, components: {sharpness, glare, aspect, complexity, blur, confidence})
  ↓
ScoredCandidate(detection_id, timestamp_ms, image_path, score)
```

### Storage

- **SQLite Database:** videos, detections, saved_cards, reviews
- **File System:** crops (all detections), frames (candidates), best (final selection)
- **Temporary:** GPU memory for batch processing

---

## GPU Acceleration Status

### Current GPU Usage

- **Variance computation:** Vectorized PyTorch (efficient)
- **Laplacian sharpness:** Vectorized PyTorch (efficient)
- **Motion/edge metrics:** Vectorized PyTorch (experimental)

### Underutilized Opportunities

1. **Batch Detector Inference:** YOLO supports batch processing (64-128 frames)
   - Currently: 1 frame at a time
   - Potential: 10-50x speedup

2. **Frame Caching:** Read full-res frames once, reuse
   - Currently: Seek for every quality check
   - Potential: Eliminate 70% of frame I/O

3. **Streaming Pipeline:** Parallel sampler + detector + scorer
   - Currently: Sequential
   - Potential: Utilize multi-core

4. **Adaptive GPU Memory:** Dynamic batch sizing
   - Currently: Fixed batch size
   - Potential: Use available VRAM more aggressively

---

## Testing & Validation

### Test Coverage
- 82 unit tests across pipeline, sampler, scoring, detector, storage
- Tests cover:
  - Frame sampling logic
  - GPU metric computation
  - Scoring accuracy
  - Selector deduplication
  - Database operations

### Manual Testing
- Tested on single 60-second lightbox video
- ~4-5 cards detected out of 7 expected
- Timing: 12-13 minutes on M2 Mac

### Gaps
- No test videos with known ground truth
- No benchmarks for detection accuracy at different qualities
- No performance regression tests

---

## Development Roadmap (Proposed)

### Phase 1: Accuracy Improvements
- [ ] Fine-tune YOLO on trading card dataset (if available)
- [ ] Add content-based deduplication (perceptual hashing)
- [ ] Implement adaptive contrast threshold per video
- [ ] Add diagnostic output (variance/motion/edge distributions)

### Phase 2: Performance Optimization
- [ ] Batch YOLO inference (10x potential speedup)
- [ ] Frame caching and memory-mapped video I/O
- [ ] Parallel processing pipeline (sampler + detector + scorer)
- [ ] Consider streaming sampler (don't store all windows in memory)

### Phase 3: Robustness
- [ ] Handle videos with varying lighting conditions
- [ ] Support different card types and orientations
- [ ] Adaptive parameter tuning per video characteristics
- [ ] Better error handling and diagnostics

---

## Questions for Architectural Review

### 1. **Detection Strategy**
   - Should we fine-tune YOLO on trading cards, or explore alternatives (R-CNN, ViT-based detection)?
   - Would a cascade of detectors (coarse-to-fine) improve accuracy without sacrificing speed?
   - Could perceptual hashing reduce duplicates without manual review?

### 2. **Performance Optimization**
   - What's the best approach for batch detector inference without memory exhaustion?
   - Should we memory-map the video file instead of repeated seeking?
   - Is a streaming pipeline (producer-consumer with queues) worth the complexity?

### 3. **Sampling Strategy**
   - Current contrast-based approach is good but brittle. Would a learned sampler (classification model to detect "card present") generalize better?
   - Could we combine multiple sampling strategies (ensemble) to catch more cards?
   - Should adaptive threshold be per-video or per-frame?

### 4. **Quality Metrics**
   - Current 6-component score is hardcoded. Should weights be learned or user-configurable?
   - Are there better quality metrics specific to trading card images?
   - Could ML-based ranking (e.g., siamese network comparing crop pairs) outperform hand-crafted scores?

### 5. **Architecture Patterns**
   - Should we adopt a plugin architecture for samplers/detectors/scorers?
   - Would a configuration-driven approach (YAML/JSON) improve flexibility?
   - Should we separate inference (heavy compute) from orchestration (lightweight)?

### 6. **Testing & Validation**
   - What's a good metric for detection accuracy in this domain (precision/recall on cards, or mean average precision)?
   - Should we create synthetic test data or prioritize real-world validation?
   - How to handle ground truth annotation (user-provided labels)?

---

## Constraints & Context

- **Target Platform:** M2/M3 Mac (Apple Silicon), Linux with NVIDIA, CPU-only fallback
- **User Expectation:** 5-10 minutes for typical 60-second video
- **Input:** Mobile phone videos (H.264, often with variable quality)
- **Output:** High-quality JPEG stills (~500KB-1MB per card)
- **Deployment:** Local (no cloud), Python CLI + web UI
- **Model Size:** Must fit in GPU memory (8-16GB typical)

---

## Appendix: Code Structure

```
card_capture/
├── __init__.py
├── cli.py              # CLI entry point, argument parsing
├── models.py           # Data classes (FrameSample, CardDetection, etc.)
├── pipeline.py         # VideoProcessor orchestration
├── sampler.py          # Frame selection strategies (740 lines)
├── detector.py         # Card detection wrappers
├── cropper.py          # Bounding box → crop extraction
├── scoring.py          # Quality scoring (6 components)
├── selector.py         # Candidate deduplication & ranking
├── storage.py          # SQLite storage layer
├── gpu_utils.py        # GPU-accelerated metrics (392 lines)
├── review.py           # Web UI for review/approval
└── templates/          # HTML templates

tests/
├── test_*.py           # 82 unit tests
└── (fixtures)

docs/superpowers/
├── specs/              # Design documents
└── plans/              # Implementation plans
```

---

## Contact & Next Steps

**Requested Review Areas:**
1. Detection accuracy improvements (especially for missed cards)
2. Performance bottleneck elimination (especially frame I/O)
3. Architectural patterns for extensibility
4. Testing strategy for validation

**Expected Outcome:**
- Prioritized list of high-impact improvements
- Feasibility assessment for each
- Proposed implementation timeline

---

*Document generated for architectural review. Questions? Please refer to README.md and inline code documentation.*
