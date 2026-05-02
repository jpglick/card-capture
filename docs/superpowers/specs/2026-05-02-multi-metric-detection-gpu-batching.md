# Multi-Metric Detection & GPU-Batched Sharpness Scoring

**Date:** May 2, 2026  
**Scope:** Improve card detection accuracy and speed for ContrastBasedSampler  
**Goal:** Capture all 7 distinct trading cards (fronts + backs) with only sharp, in-focus frames, while maintaining fast processing

## Problem Statement

Current ContrastBasedSampler uses single-metric detection (Laplacian variance only):
- **Missing cards unpredictably** — some fronts/backs not detected
- **Slow Pass 2 bottleneck** — sharpness scoring takes ~223s for a 34s video, limiting flexibility for threshold tuning
- **Limited adaptability** — cards with low contrast or unusual texture patterns fall below threshold

## Solution Overview

Two complementary improvements:

**A) Multi-Metric Detection** — Add 3 lightweight GPU detectors alongside Laplacian variance:
- Motion detection (catch cards being positioned)
- Histogram outlier detection (catch statistically unusual variance frames)
- Edge density detection (catch textured cards)
- Fusion via logical OR (any detector triggering = "card present")
- Pass 2 sharpness filtering removes blurry frames

**B) GPU-Batched Sharpness Scoring** — Replace sequential frame processing with parallel batch processing:
- Process 32-50 frames at once on GPU
- Expected 3.7-5.6x speedup for Pass 2
- Enable lower thresholds without time penalty

## Detailed Design

### A) Multi-Metric Detection System

#### Architecture

```
Pass 1: Low-res video scan
├── Metric 1: Laplacian Variance (primary detector, existing)
├── Metric 2: Motion Detection (frame-to-frame pixel delta)
├── Metric 3: Histogram Outlier Detection (statistical anomaly)
├── Metric 4: Edge Density Detection (high-edge regions)
└── Fusion: OR all metrics → PresenceWindow

Pass 2: Full-res sharpness scoring (GPU-batched)
└── Sort by sharpness, select top N candidates
```

#### Metric 1: Laplacian Variance (Existing, Enhanced)

**No changes to existing logic.** Remains primary detector.
- Laplacian kernel measures edge/texture density
- GPU-accelerated via PyTorch (existing)
- Current threshold: 600.0

#### Metric 2: Motion Detection

**Purpose:** Catch cards being positioned (higher motion = likely card being moved/placed).

**Algorithm:**
1. Compute grayscale absolute difference between consecutive downscaled frames
2. Compute mean pixel delta across image: `mean_delta = mean(|frame[t] - frame[t-1]|)`
3. Threshold: `motion_detected = mean_delta > motion_threshold` (default: 8.0, tunable)
4. Run on GPU: `torch.abs(frame1_tensor - frame2_tensor).mean()`

**Rationale:**
- Cards held still have low motion; cards being positioned have higher motion
- Captures transition moments where card presence begins
- Orthogonal to variance (motion ≠ texture)
- Negligible GPU overhead (single tensor operation)

**Parameter:** `--motion-threshold` (default: 8.0, range 1-50)
- Higher values: only large movements (fewer detections, lower false positives)
- Lower values: catch all motion (more detections, may catch camera shake)

#### Metric 3: Histogram Outlier Detection

**Purpose:** Catch statistically unusual frames (dim cards, bright reflections, unusual lighting).

**Algorithm:**
1. During Pass 1, collect Laplacian variance values for all scanned frames into histogram
2. Compute mean (μ) and standard deviation (σ) of variances
3. Mark frame as "outlier" if variance is outside band: `|variance - μ| > k·σ` (default k=1.5)
4. Rationale: Unusual cards (very dim, very bright, or unusual texture) deviate from typical lightbox variance

**Rationale:**
- Lightbox typically has consistent variance pattern (empty frames cluster around μ)
- Cards represent outliers: they're either significantly higher (high-contrast cards) or lower (dim cards) than the norm
- k=1.5 is sensitive (catches ±1.5σ deviations); tunable for more/less sensitivity

**Parameter:** `--histogram-outlier-sigma` (default: 1.5, range 0.5-3.0)
- Lower values: more sensitive (larger ±σ band), catches more outliers
- Higher values: less sensitive, only extreme outliers trigger

#### Metric 4: Edge Density Detection

**Purpose:** Catch textured/patterned cards that may have lower overall variance but high local edge density.

**Algorithm:**
1. Compute Sobel edge magnitude at low resolution (same as scan image)
2. Compute percentage of high-edge pixels: `edge_pixels = count(|sobel| > edge_threshold) / total_pixels` (default edge_threshold: 50)
3. Threshold: `high_edges_detected = edge_pixels > edge_density_threshold` (default: 0.15, i.e., >15% edge pixels)
4. Run on GPU: `torch.nn.functional.conv2d` with Sobel kernels

**Rationale:**
- Textured/patterned cards have high edge content but may have low overall variance
- Orthogonal to variance metric
- Catches metallic/holographic cards, printed patterns, fine details

**Parameters:**
- `--edge-density-threshold` (default: 0.15, range 0.05-0.50)
  - Higher values: only very high-edge frames (fewer detections)
  - Lower values: catch any textured content (more detections)
- `--sobel-magnitude-threshold` (default: 50, range 20-150)
  - Higher values: only strong edges count (fewer edge pixels)
  - Lower values: count weaker edges (more edge pixels)

#### Metric Fusion

**Logical OR:** A frame is marked "card present" if ANY metric triggers:

```python
card_present = (
    variance > contrast_threshold OR
    motion > motion_threshold OR
    histogram_outlier OR
    edge_density > edge_density_threshold
)
```

**Rationale:**
- Maximizes sensitivity: catch all possible cards
- Pass 2 sharpness filtering removes blurry frames (false positives)
- No false negatives: if any detector sees a card, it's captured

**Trade-off:** May increase false positives (e.g., lightbox edge reflections), but Pass 2 filters them out.

#### Detection Method Tracking

**New field in PresenceWindow:**
```python
@dataclass
class PresenceWindow:
    start_frame: int
    end_frame: int
    frame_candidates: list[tuple[int, float]]  # (frame_index, sharpness_score)
    detection_methods: list[str]  # which metrics fired: ["variance", "motion", "histogram"]
```

**Purpose:** Track which metric(s) detected each window for debugging/optimization.

#### Configuration

**CLI flag:** `--detection-metrics` (comma-separated list, default: "variance,motion,histogram,edges")

```bash
card-capture process video.mov \
  --sampler contrast \
  --detection-metrics variance,motion,histogram,edges \
  --contrast-threshold 600.0 \
  --motion-threshold 8.0 \
  --histogram-outlier-sigma 1.5 \
  --edge-density-threshold 0.15 \
  --sobel-magnitude-threshold 50
```

### B) GPU-Batched Sharpness Scoring

#### Current Bottleneck

Pass 2 (sharpness scoring) processes frames sequentially:
- ~223s for ~34s video = 6.6x real-time
- One frame at a time via `cv2.Laplacian()`
- GPU underutilized (dispatch overhead > compute cost)

#### New Approach

**Batch-process frames on GPU:**

1. **Collect all candidate frames** from presence window
2. **Form batches** of N frames (default: 32, tunable: 1-128)
3. **Load batch to GPU:** Stack frames into single tensor
4. **Compute sharpness in parallel:** `torch.nn.functional.conv2d` on entire batch
5. **Extract scores** and sort by sharpness
6. **Select top K candidates**

#### Algorithm

```python
def score_sharpness_batched(frames, batch_size=32):
    """Score sharpness for multiple frames in parallel batches."""
    all_scores = []
    
    for batch_start in range(0, len(frames), batch_size):
        batch_end = min(batch_start + batch_size, len(frames))
        batch = frames[batch_start:batch_end]
        
        # Stack batch into single tensor (B, H, W, C)
        batch_tensor = torch.stack([torch.from_numpy(f) for f in batch])
        batch_tensor = batch_tensor.to(device).float()
        
        # Convert to grayscale and compute Laplacian variance
        gray = torch.mean(batch_tensor, dim=3)  # (B, H, W)
        gray = gray.unsqueeze(1)  # (B, 1, H, W) for conv2d
        
        # Laplacian kernel
        laplacian_kernel = torch.tensor([...]).to(device)
        edges = torch.nn.functional.conv2d(gray, laplacian_kernel)
        
        # Compute variance across spatial dimensions
        batch_scores = torch.var(edges, dim=(2, 3))  # (B,)
        all_scores.extend(batch_scores.cpu().numpy())
    
    return all_scores
```

#### Memory Management

**GPU VRAM budgeting:**

- M2 Max (16GB): 50-100 frames per batch (typical ~150MB per 50 frames)
- M2 Pro (8GB): 30-50 frames per batch (typical ~100MB per 50 frames)
- GPU < 4GB: Fall back to batch_size=1 (sequential, like current)

**Auto-detection:**
```python
def estimate_batch_size(device, frame_shape=(1080, 1920), bytes_per_frame_factor=3):
    """Estimate safe batch size based on available VRAM."""
    if device.type == "cpu":
        return 1  # CPU: sequential only
    
    available_vram = torch.cuda.get_device_properties(device).total_memory
    bytes_per_frame = frame_shape[0] * frame_shape[1] * bytes_per_frame_factor
    safe_batch_size = max(1, (available_vram * 0.5) // bytes_per_frame)
    
    return min(safe_batch_size, 128)  # Cap at 128
```

#### Expected Speedup

**Baseline Pass 2:** ~223s (sequential frame processing)

**Batched Pass 2:**
- Batch_size=32: ~60-80s (2.8-3.7x speedup)
- Batch_size=50: ~40-60s (3.7-5.6x speedup)
- Batch_size=100: ~30-50s (4.5-7.4x speedup, if VRAM allows)

**Assumptions:**
- GPU kernel dispatch overhead amortized across batch
- PyTorch batched operations highly optimized
- No frame load/unload delays between batches

#### Configuration

**CLI flag:** `--sharpness-batch-size` (default: 32, range: 1-128)

```bash
card-capture process video.mov \
  --sampler contrast \
  --sharpness-batch-size 50
```

**Auto-detect best batch size:** `--sharpness-batch-size auto` (queries available VRAM)

#### Backward Compatibility

- Default batch_size=32 (works on M2 Pro with 8GB)
- batch_size=1 equivalent to current sequential behavior
- CPU-only falls back to batch_size=1 automatically
- Existing tests pass without modification (old behavior still supported)

## Integration

### Modified Components

**src/card_capture/sampler.py:**
- `ContrastBasedSampler.__init__()`: Add parameters for new metrics
- `ContrastBasedSampler._find_presence_windows()`: Implement metric fusion logic
- `ContrastBasedSampler._score_sharpness_in_window()`: Replace with `_score_sharpness_batched()`
- `PresenceWindow` dataclass: Add `detection_methods` field

**src/card_capture/gpu_utils.py:**
- Add `compute_motion_gpu()` function
- Add `compute_histogram_outliers_gpu()` function
- Add `compute_edge_density_gpu()` function
- Add `estimate_batch_size()` function
- Add `score_sharpness_batched()` function

**src/card_capture/cli.py:**
- Add `--detection-metrics` flag
- Add `--motion-threshold` flag
- Add `--histogram-outlier-sigma` flag
- Add `--edge-density-threshold` flag
- Add `--sobel-magnitude-threshold` flag
- Add `--sharpness-batch-size` flag

**README.md:**
- Document new metrics and parameters
- Add examples for different tuning scenarios

### New Tests

**tests/test_gpu_utils.py:**
- Test `compute_motion_gpu()` with known motion scenarios
- Test `compute_histogram_outliers_gpu()` with outlier detection
- Test `compute_edge_density_gpu()` with textured images
- Test `estimate_batch_size()` with different VRAM scenarios
- Test `score_sharpness_batched()` correctness vs sequential

**tests/test_sampler.py:**
- Test metric fusion logic (any metric triggering marks frame as present)
- Test `detection_methods` tracking in PresenceWindow
- Test batched sharpness scoring produces same top-K candidates as sequential
- Test CLI parameter wiring for new flags

## Success Criteria

✅ **Accuracy:**
- Detect all 7 trading cards (fronts + backs) from test video
- Zero false positives in output (only sharp, in-focus frames)
- No regression on existing single-metric tests

✅ **Performance:**
- Pass 2 sharpness scoring: < 70 seconds (from 223s baseline)
- Total processing: < 150 seconds (from 288s baseline)
- Timing output shows contribution of each metric

✅ **Usability:**
- All new parameters have sensible defaults
- Auto-detection of batch size and device works reliably
- CLI help documents all new flags
- README provides tuning guidance

✅ **Testing:**
- 55+ existing tests pass (no regressions)
- 15+ new tests for multi-metric and batching
- Manual verification on real video (IMG_5596.MOV)

## Timeline & Phases

**Phase 1:** GPU utilities for new metrics (2-3 tasks)
- Implement motion, histogram, edge density detectors with tests

**Phase 2:** Integration into ContrastBasedSampler (2-3 tasks)
- Metric fusion logic, PresenceWindow updates, timing output

**Phase 3:** Batched sharpness scoring (1-2 tasks)
- Implement `score_sharpness_batched()`, memory management, tests

**Phase 4:** CLI integration & documentation (2 tasks)
- Wire flags, update README, examples

**Phase 5:** Validation & optimization (1 task)
- Real video testing, performance profiling, tuning recommendations
