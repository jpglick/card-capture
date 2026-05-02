# Quick Reference: Card Capture Architecture

## Problem Summary
Extract high-quality trading card images from hand-held lightbox videos. Currently achieving ~70% detection accuracy on test videos with 12-13 minute processing time for 60-second videos.

## High-Level Pipeline
```
Video → Sampler (find card windows) → Detector (YOLO) → Scorer (6 metrics) → Selector (pick best) → Output
```

## Key Metrics
- **Detection Accuracy:** ~70% (missing 2-4 cards per 10-card video)
- **Processing Speed:** 12-13 minutes for 60-second video (M2 Mac)
- **Time Breakdown:** 
  - Pass 1 (contrast scan): ~60s
  - Pass 2 (sharpness scoring): ~180-600s (frame seeking bottleneck)
  - Detection/scoring: ~30s

## Core Components

| Component | Purpose | Lines | Status |
|-----------|---------|-------|--------|
| **Sampler** | Frame selection strategy | 745 | Working (ContrastBased best) |
| **Detector** | Card localization (YOLO) | 139 | Working but low accuracy |
| **Scorer** | Quality assessment (6 metrics) | 58 | Working |
| **Selector** | Deduplication & ranking | 39 | Working |
| **GPU Utils** | Metric computation | 392 | Working (underutilized) |
| **Pipeline** | Orchestration | 148 | Working |

## Sampler Strategies

1. **VideoSampler** — Every Nth frame
2. **StabilityBasedSampler** — Low-motion windows
3. **DetectionGuidedSampler** — YOLO-detected windows
4. **ContrastBasedSampler** — High-variance windows (✓ Recommended)
   - Pass 1: Scan at low-res, find variance > threshold
   - Pass 2: Within windows, rank by Laplacian sharpness

## Top Performance Bottlenecks

1. **Frame Seeking** (~40s per frame read)
   - Currently: Seek → read → decode → process for each candidate
   - Potential Fix: Memory-map video file or batch frame reading

2. **Single-Frame Detector Inference** (~1-2s per frame)
   - Currently: YOLO inference on 1 frame at a time
   - Potential Fix: Batch inference (64-128 frames) = 10-50x speedup

3. **Sequential Sampler** (~5-10m of 12-13m total)
   - Currently: Pass 1 + Pass 2 sequential
   - Potential Fix: Streaming/pipelined sampler

## Top Detection Accuracy Issues

1. **YOLO Not Trained on Trading Cards**
   - Misses cards at angles, with reflections, overlaps
   - Potential Fix: Fine-tune on trading card dataset

2. **Parameter Tuning Required**
   - contrast_threshold: 600-3000 depending on video
   - Potential Fix: Adaptive threshold per video

3. **Duplicate Detection** (occasional)
   - Time-window grouping can merge distinct cards
   - Potential Fix: Content-based deduplication

## Questions for LLM Review

**Accuracy:**
- Best approach to improve YOLO performance on trading cards?
- Should we use a cascade (coarse-to-fine) or ensemble?
- Could perceptual hashing eliminate duplicates?

**Performance:**
- Best way to batch YOLO inference without memory issues?
- Memory-mapped I/O vs. buffered caching for video?
- Worth implementing streaming/pipelined architecture?

**Architecture:**
- Plugin architecture for samplers/detectors/scorers?
- Configuration-driven (YAML) vs. programmatic?
- How to decouple inference from orchestration?

**Testing:**
- What metrics for detection accuracy in this domain?
- Synthetic vs. real-world validation approach?
- How to handle ground truth annotation?

## File Locations

- **Main Code:** `/Users/josh/WebstormProjects/vc2/src/card_capture/`
- **Full Review:** `ARCHITECTURE_REVIEW_REQUEST.md` (462 lines, comprehensive)
- **Tests:** `/Users/josh/WebstormProjects/vc2/tests/` (82 tests, passing)

## Current Test Results

```
82 tests passing ✓
- Pipeline: 4 tests
- Sampler: 30+ tests (ContrastBased sampler tests included)
- Scoring/Selection: 8 tests
- Storage: 2 tests
- Detector: 5 tests
- GPU Utils: 8 tests
- CLI: 3 tests
```

## Usage Examples

### Basic Processing
```bash
PYTHONPATH=src python3 -m card_capture.cli process \
  ~/video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite
```

### With Contrast Sampler (Recommended)
```bash
PYTHONPATH=src python3 -m card_capture.cli process \
  ~/video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite \
  --sampler contrast \
  --contrast-threshold 2000.0
```

### With GPU Acceleration
```bash
PYTHONPATH=src python3 -m card_capture.cli process \
  ~/video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite \
  --sampler contrast \
  --device mps  # Apple Silicon
  # or --device cuda for NVIDIA
```

## Next Steps

1. Share full `ARCHITECTURE_REVIEW_REQUEST.md` with LLM for architectural review
2. Request prioritized list of high-impact improvements
3. Feasibility assessment for performance + accuracy gains
4. Implementation timeline for Phase 1 (accuracy improvements)
