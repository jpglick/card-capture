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
1. Compute motion between consecutive downscaled frames (same resolution as Pass 1 scan: `scan_width` pixels wide, e.g., 160px)
2. Compute grayscale absolute difference: `motion_frame = abs(gray_frame[t] - gray_frame[t-1])`
3. Compute mean pixel delta across entire image: `mean_delta = mean(motion_frame)`
4. Threshold: `motion_detected = mean_delta > motion_threshold` (default: 8.0)
5. Run on GPU: `torch.abs(frame1_tensor - frame2_tensor).mean()`

**Rationale:**
- Cards held still have low motion; cards being positioned have higher motion
- Captures transition moments where card presence begins
- Orthogonal to variance (motion ≠ texture)
- Negligible GPU overhead (single tensor operation)
- Motion delta is normalized to 0-255 range (for uint8 grayscale), so threshold of 8.0 means ~3% pixel change

**Parameter:** `--motion-threshold` (default: 8.0, range 1-50)
- **Higher values (e.g., 20+):** Only large movements trigger; ignores small jitter and focus breathing
- **Lower values (e.g., 3-5):** Catch all motion; may trigger on camera shake or lighting flicker
- **Default 8.0:** Catches card positioning (typically 10-30% pixel change) while ignoring minor jitter

#### Metric 3: Histogram Outlier Detection

**Purpose:** Catch statistically unusual frames (dim cards, bright reflections, unusual lighting).

**Algorithm (Two-Phase Pass 1):**

*Phase 1A - Statistics Collection:*
1. Scan entire video, compute Laplacian variance for all frames at scan resolution
2. Collect all variance values into histogram
3. Compute mean (μ) and standard deviation (σ) of all variances

*Phase 1B - Outlier Detection (with Fusion):*
4. Scan video again with metric fusion
5. For each frame, mark as "outlier" if: `|variance - μ| > k·σ` (default k=1.5)
6. Interpretation: Frames with unusually HIGH variance (card/texture) OR unusually LOW variance (rare but possible with specific lighting)

**Rationale:**
- Lightbox typically has consistent variance pattern (empty frames cluster around μ = baseline)
- Cards represent outliers: they deviate from the typical lightbox variance distribution
- Higher k values = stricter threshold (only extreme outliers detected, fewer detections)
- Lower k values = looser threshold (more frames qualify as outliers, catches subtle variations)

**Parameter:** `--histogram-outlier-sigma` (default: 1.5, range 0.5-3.0)
- **1.5 (default):** Moderate sensitivity; catches cards that are ±1.5σ from baseline
- **Lower values (e.g., 0.5):** More sensitive; catches cards closer to baseline variance
- **Higher values (e.g., 2.5):** Less sensitive; only extreme outliers trigger

**Implementation Note:** This metric requires two passes through the video (stats collection, then detection). The overhead is acceptable since Pass 1 is already fast (~65s). Consider caching histogram statistics if the same video is processed multiple times.

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

**CLI flag:** `--detection-metrics` (comma-separated list, default: "variance")

```bash
# Default (backward compatible): only Laplacian variance
card-capture process video.mov --sampler contrast

# All metrics enabled (recommended for challenging videos)
card-capture process video.mov \
  --sampler contrast \
  --detection-metrics variance,motion,histogram,edges \
  --motion-threshold 8.0 \
  --histogram-outlier-sigma 1.5 \
  --edge-density-threshold 0.15

# Custom mix
card-capture process video.mov \
  --sampler contrast \
  --detection-metrics variance,motion \
  --motion-threshold 5.0
```

**Default Behavior (Backward Compatible):**
- `--detection-metrics variance` (only existing Laplacian variance)
- Existing tests continue to pass without modification
- New metrics must be explicitly enabled by user

**Progressive Enhancement:**
- Start with default (variance only)
- If cards are missed, enable `--detection-metrics variance,motion` to catch positioning transitions
- If still missing cards, enable histogram: `--detection-metrics variance,motion,histogram`
- If textured cards are missed, add edges: `--detection-metrics variance,motion,histogram,edges`

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
def score_sharpness_batched(frames, batch_size=32, device="mps"):
    """Score sharpness for multiple frames in parallel batches.
    
    Args:
        frames: List of numpy arrays (H, W, C) or (H, W)
        batch_size: Number of frames to process at once
        device: torch device (mps, cuda, cpu)
    
    Returns:
        List of sharpness scores (one per frame)
    """
    # Laplacian kernel (3x3) for edge detection
    laplacian_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32).to(device)
    laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    
    all_scores = []
    
    for batch_start in range(0, len(frames), batch_size):
        batch_end = min(batch_start + batch_size, len(frames))
        batch = frames[batch_start:batch_end]
        
        # Stack batch into single tensor (B, H, W, C)
        batch_tensor = torch.stack([torch.from_numpy(f) for f in batch])
        batch_tensor = batch_tensor.to(device).float()
        
        # Convert to grayscale and prepare for conv2d
        if batch_tensor.shape[-1] == 3:  # RGB
            gray = torch.mean(batch_tensor, dim=3)  # (B, H, W)
        else:  # Already grayscale
            gray = batch_tensor.squeeze(-1)
        
        gray = gray.unsqueeze(1)  # (B, 1, H, W) for conv2d
        
        # Apply Laplacian kernel (edges)
        edges = torch.nn.functional.conv2d(gray, laplacian_kernel, padding=1)  # (B, 1, H, W)
        
        # Compute variance across spatial dimensions (sharpness = high variance of edge magnitudes)
        batch_scores = torch.var(edges, dim=(2, 3)).squeeze()  # (B,)
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
def estimate_batch_size(device, frame_shape=(1080, 1920), safety_margin=0.4):
    """Estimate safe batch size based on available VRAM.
    
    Accounts for: frame data, intermediate Laplacian tensors, variance computations.
    Uses conservative estimate to avoid OOM.
    """
    if device.type == "cpu":
        return 1  # CPU: sequential only
    
    available_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    
    # Empirical estimate: ~10MB per frame for batched Laplacian + variance
    # (includes overhead for intermediate tensors and PyTorch memory fragmentation)
    mb_per_frame = 10
    usable_vram_gb = available_vram_gb * (1 - safety_margin)  # Leave 40% free
    usable_vram_mb = usable_vram_gb * 1024
    
    safe_batch_size = int(usable_vram_mb / mb_per_frame)
    
    return min(max(1, safe_batch_size), 128)  # Clamp to [1, 128]
```

**VRAM Guidelines (Empirical):**
- M2 Max (16GB): ~110 frames per batch (~15GB usable after 40% margin)
- M2 Pro (8GB): ~50 frames per batch (~4.8GB usable)
- Default 32: Safe on most recent consumer GPUs (<=8GB)
- If OOM occurs: Reduce `--sharpness-batch-size` manually or enable auto-detect

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

- **Default detection metrics:** `variance` only (existing behavior preserved)
  - All existing tests pass without modification
  - New metrics must be explicitly enabled via `--detection-metrics` flag
- **Default batch_size:** 32 (safe on machines with 8GB+ VRAM)
  - batch_size=1 equivalent to current sequential behavior if needed
  - CPU-only falls back to batch_size=1 automatically
- **Existing parameters unchanged:** All current CLI flags and defaults remain the same
- **Progressive adoption:** Users can opt-in to new metrics as needed for better detection

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
- Detect all 7 trading cards (fronts + backs) from test videos
- Zero false positives in output (only sharp, in-focus frames)
- No regression on existing single-metric tests (variance-only baseline behavior maintained)
- Multi-metric mode improves detection rate over baseline

✅ **Performance:**
- Pass 2 sharpness scoring: < 70 seconds (from 223s baseline, 3.2x speedup)
- Total processing: < 150 seconds (from 288s baseline, 1.9x speedup)
- Timing output shows breakdown of each stage

✅ **Usability:**
- Default detection mode is backward compatible (variance only)
- All new parameters have sensible defaults
- Auto-detection of batch size and device works reliably
- CLI help documents all new flags
- README provides tuning guidance and examples

✅ **Testing:**
- 55+ existing tests pass (no regressions with variance-only default)
- 15+ new tests for multi-metric and batching
- Manual verification on test videos (if available)

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
