# ContrastBasedSampler: Card-Presence-Aware Frame Selection

**Date:** 2026-05-01  
**Status:** Design Review  
**Author:** Copilot

## Problem Statement

Current frame sampling strategies (motion-based `StabilityBasedSampler` and detection-based `DetectionGuidedSampler`) fail to reliably select high-quality card images from hand-held lightbox videos. The core issue:

- **Motion-based:** Picks the *most still* moments, which are often empty lightbox frames (no card)
- **Detection-based:** YOLO model misses cards and produces false positives (lightbox edges)

User's workflow: hand-place cards under camera in a lightbox, hold briefly, remove. The plain lightbox background provides natural visual contrast with any card (brightly colored or detailed). Goal: extract the sharpest frame(s) when a card is actually present in frame.

## Solution Overview

**ContrastBasedSampler:** Two-pass sampler that leverages high contrast between cards and plain lightbox background (no ML required).

- **Pass 1:** Scan video at low resolution, detect card presence by color variance (anything with variance > threshold is "not plain lightbox")
- **Pass 2:** Within presence windows, compute sharpness and yield the sharpest frame(s)

Result: Reliable, fast frame selection that naturally finds motion-free moments when cards are clearly visible.

## Design

### Pass 1: Contrast-Based Presence Detection

**Input:** Video file, scan parameters

**Process:**
1. Decode video at `scan_fps` (default 5 fps)
2. Downscale each frame to `scan_width` pixels wide (default 160)
3. For each frame, compute **color variance** across RGB channels:
   - Convert frame to RGB
   - Compute mean of all pixel channel values
   - Compute variance: `mean((pixel - mean)^2)` across all pixels and channels
   - This metric is high when frame contains diverse colors/patterns (card present)
   - This metric is low when frame is mostly uniform white (empty lightbox)
4. Track consecutive frames with variance > `contrast_threshold` (default 1000)
5. Merge adjacent high-variance frames into presence windows
6. Discard windows with fewer than `min_presence_frames` (default 3)

**Output:** List of `PresenceWindow` objects, each with:
- `start_frame`: first frame index in window
- `end_frame`: last frame index in window
- `frame_samples`: list of `(frame_index, variance)` tuples for all frames in window

**Why variance works:**
- Plain white lightbox ≈ low variance
- Card (any color, any pattern) ≈ high variance
- No color-specific tuning needed
- Robust to lighting changes (uses relative contrast, not absolute RGB values)
- Fast computation (no ML, no complex algorithms)

### Pass 2: Sharpness-Based Selection

**Input:** List of presence windows, full-resolution video

**Process:**
1. For each presence window:
   - Decode full-resolution frames at each sample point
   - Compute Laplacian variance (sharpness metric) for each frame
   - Sort by sharpness descending
   - Select top `candidates_per_window` frames (default 3)
   - Yield each as a `FrameSample` to the pipeline

**Output:** `Iterator[FrameSample]` — candidate frames from presence windows, in order

**Why Laplacian variance works:**
- Measures high-frequency content (edge sharpness)
- Peaks when card is still and in focus
- Naturally selects the best moments within a presence window
- Existing proven heuristic in pipeline

### Parameters

| Parameter | Default | Type | Pass | Description |
|-----------|---------|------|------|-------------|
| `scan_fps` | 5 | float | 1 | Frames per second to scan at |
| `scan_width` | 160 | int | 1 | Downscale width for scanning (pixels) |
| `contrast_threshold` | 1000 | float | 1 | Variance threshold to detect card presence |
| `min_presence_frames` | 3 | int | 1 | Minimum consecutive frames to form a presence window |
| `candidates_per_window` | 3 | int | 2 | Number of sharpest frames to yield per window |

### New Dataclass: PresenceWindow

```python
@dataclass
class PresenceWindow:
    """A contiguous run of high-contrast (card-present) frames.
    
    start_frame: First frame index in window (source video frame number)
    end_frame: Last frame index in window (source video frame number)
    frame_samples: List of (frame_index, variance) for all frames in window
    """
    start_frame: int
    end_frame: int
    frame_samples: List[Tuple[int, float]]
```

### New Class: ContrastBasedSampler

**Interface:** Implements sampler protocol (has `sample(video_path, sample_fps)` method)

**Constructor:**
```python
def __init__(
    self,
    scan_fps: float = 5.0,
    scan_width: int = 160,
    contrast_threshold: float = 1000.0,
    min_presence_frames: int = 3,
    candidates_per_window: int = 3,
) -> None
```

**Methods:**
- `_find_presence_windows(video_path: Path) -> List[PresenceWindow]` — Pass 1 implementation
- `sample(video_path: Path, sample_fps: float) -> Iterator[FrameSample]` — Pass 2 implementation

**Note:** `sample_fps` parameter is ignored (sampler uses `scan_fps` only, for interface compatibility with `VideoSampler`).

### CLI Integration

**New sampler choice:**
```bash
--sampler contrast  # ContrastBasedSampler (new default)
```

**New flags:**
```bash
--contrast-threshold FLOAT            # Variance threshold (default: 1000.0)
--min-presence-frames INT             # Min consecutive frames (default: 3)
```

**Existing flags reused:**
```bash
--scan-fps FLOAT                      # Pass 1 scan cadence (default: 5.0)
--scan-width INT                      # Pass 1 downscale width (default: 160)
--candidates-per-window INT           # Pass 2 candidates per window (default: 3)
```

**Example command:**
```bash
PYTHONPATH=src python3 -m card_capture.cli process video.mov \
  --sampler contrast \
  --scan-fps 3 \
  --contrast-threshold 1200 \
  --candidates-per-window 5 \
  --detections-to-stop 0
```

### Testing Strategy

**Unit tests for Pass 1:**
- Single card in frame → single presence window detected
- Empty lightbox → no presence window
- Multiple distinct cards → multiple windows
- Card motion through frame → single long window (not multiple windows)
- Contrast threshold tuning: frames at/above/below threshold handled correctly
- Window merging: adjacent high-variance frames correctly merged

**Unit tests for Pass 2:**
- Sharpness ranking: sharper frames score higher
- Candidate distribution: top N frames selected and spread across window
- Full pipeline: end-to-end from video to `FrameSample` yields

**Integration tests:**
- Realistic lightbox video with cards
- Compare results to visual inspection

### Error Handling

**Pass 1 errors:**
- Video file not readable → raise `ValueError` (consistent with other samplers)
- Video has no frames → return empty list (no windows found)

**Pass 2 errors:**
- Frame decode failure → skip frame, continue to next
- Empty window list → yield nothing (no candidates)

### Performance

**Expected performance:**
- Pass 1: O(n) where n = number of scanned frames (~video_duration_seconds * scan_fps)
- Per-frame: variance computation is O(pixels), fast (downscaled to 160px wide)
- Pass 2: O(m * k * log k) where m = number of windows, k = frames per window
- Laplacian computation is standard and O(pixels)

For a 34-second video at 5 fps scan:
- Pass 1: ~170 frames to scan, ~0.5s total (10ms per frame at 160px)
- Pass 2: ~8 windows * 3 candidates * full-res decode + Laplacian = ~2-3s

**Total: ~3-4 seconds** (vs ~90s for YOLO-based approach)

### Relationship to Existing Code

**Sampler interface:** `ContrastBasedSampler` follows same interface as `StabilityBasedSampler` and `VideoSampler`

**Dataclasses:** New `PresenceWindow` parallel to existing `StableWindow` and `DetectionWindow`

**Pipeline integration:** Works with existing `VideoProcessor`, no changes needed

**CLI:** Extends existing sampler choices, no breaking changes

### Deprecation

After `ContrastBasedSampler` is proven on real videos:
- Keep `StabilityBasedSampler` as fallback (may be useful for other video types)
- Consider deprecating `DetectionGuidedSampler` (YOLO-based didn't work)
- Update default sampler from "stability" to "contrast"

## Success Criteria

1. **Correctness:** Selects frames where cards are actually present (no more empty lightbox frames)
2. **Quality:** Picks sharp frames (Laplacian-based sharpness selection)
3. **Speed:** Pass 1 + Pass 2 complete in <5 seconds for 34-second video
4. **Robustness:** Works across diverse card colors and patterns without tuning
5. **User experience:** Default parameters work for most lightbox setups without adjustment

## Open Questions / Future Refinement

- **Contrast threshold tuning:** Default 1000 based on typical 160px downscaled frames. May need user exposure/adjustment flag if some lightboxes are off-white or tinted.
- **Window merging:** Currently treat all high-variance regions as presence. Could add temporal gap tolerance (e.g., merge windows separated by <0.5s of low variance) to handle brief camera shake during card insertion.
- **Adaptive thresholds:** Could compute threshold per-video by analyzing histogram of frame variances (auto-calibrate to user's lightbox).

## Implementation Phases

**Phase 1:** Implement `ContrastBasedSampler` with Pass 1 and Pass 2, add CLI flags, write tests, verify on real video

**Phase 2 (future):** Optimize based on real-world feedback (threshold tuning, window merging, etc.)
