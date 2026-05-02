# Card Capture

Extract high-quality sports card stills from local video files.

## Installation

```bash
pip install -e .
```

To use the ML-based card detection features, also install the optional model dependencies:

```bash
pip install -e ".[model]"
```

## Quick Start

### Process a Video

Extract card stills from a video file using the default stability-based sampler:

```bash
card-capture process ~/path/to/video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite
```

The extracted frames are saved to the output directory, and metadata is stored in the SQLite database.

### Using the Contrast-Based Sampler (Recommended for Lightbox Videos)

For videos taken in a controlled lightbox environment with manual card placement, the contrast-based sampler provides fast, ML-free frame selection with GPU acceleration:

```bash
PYTHONPATH=src python3 -m card_capture.cli process ~/path/to/video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite \
  --sampler contrast \
  --contrast-threshold 600.0 \
  --min-presence-frames 3 \
  --candidates-per-window 3 \
  --window-merge-gap 5 \
  --device auto
```

Or using the installed command:

```bash
card-capture process ~/path/to/video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite \
  --sampler contrast \
  --contrast-threshold 600.0 \
  --min-presence-frames 3 \
  --candidates-per-window 3 \
  --window-merge-gap 5 \
  --device auto
```

**Key parameters:**
- `--contrast-threshold`: Minimum color variance (higher = more selective). Default: 600.0
- `--min-presence-frames`: Min consecutive frames for a presence window. Default: 3
- `--candidates-per-window`: Sharpest frames per window. Default: 3
- `--window-merge-gap`: Frame gap tolerance for merging jittery windows. Default: 5 (NEW)
- `--device`: GPU acceleration ("auto" detects best, "cpu" for CPU-only, "mps" for Apple Silicon). Default: auto (NEW)

#### How It Works

The contrast-based sampler uses a two-pass algorithm:

1. **Pass 1 (Detection):** Scans the video at low resolution to detect card presence via color variance. Frames are scanned at the rate specified by `--scan-fps` and resized to `--scan-width` pixels. Frames with variance exceeding `--contrast-threshold` are marked as "card present."

2. **Pass 2 (Selection):** Within each window of consecutive "card present" frames, ranks all frames by Laplacian sharpness and selects the N sharpest as candidates.

#### Parameters

- `--contrast-threshold` (default: 1000.0)
  - Minimum color variance required to detect card presence
  - Higher values are more selective (fewer false positives)
  - Lower values are more sensitive (catch more variations)
  - Adjust if detection misses cards or detects empty lightbox frames

- `--min-presence-frames` (default: 3)
  - Minimum number of consecutive frames to form a presence window
  - Filters out noise and brief transients
  - Higher values require longer card presence periods

- `--candidates-per-window` (default: 5)
  - Number of sharpest frames to yield per presence window
  - Higher values = more redundancy for manual review, longer processing
  - Lower values = faster processing, fewer options per window

- `--scan-fps` (default: 10.0)
  - Frames per second to scan in Pass 1
  - Lower values = faster Pass 1 scan but coarser detection
  - Higher values = finer detection but slower Pass 1 scan
  - Recommended: 5-10 fps for most videos

- `--scan-width` (default: 160)
  - Frame width in pixels for Pass 1 scan
  - Lower values = faster scan but may miss fine details
  - Higher values = better detail preservation but slower scan
  - Recommended: 160-320 pixels for variance-based detection

- `--window-merge-gap` (default: 5)
  - Frame gap tolerance for merging jittery presence windows
  - Handles card movement jitter during manual placement by merging windows separated by ≤ N frames
  - Lower values (2-3): Stricter merging, only combine closely-spaced windows
  - Higher values (8-10): Aggressive merging, may combine distant windows into false positives
  - Typical range: 3-7 frames

- `--device` (default: auto)
  - Device to use for GPU acceleration: `auto`, `cpu`, `mps` (Apple Silicon), or `cuda` (NVIDIA)
  - `auto`: Automatically detects and uses available GPU, falls back to CPU
  - `cpu`: Force CPU-only mode (useful for testing or GPU troubleshooting)
  - `mps`: Force Metal Performance Shaders (Apple Silicon M1/M2/M3)
  - `cuda`: Force CUDA acceleration (NVIDIA GPUs)

#### Performance Optimization

**GPU Acceleration:** The contrast-based sampler automatically detects and uses available GPU acceleration:

- **Apple Silicon (M1/M2/M3):** Uses Metal Performance Shaders (MPS) for 3-5x speedup
- **NVIDIA GPUs:** Uses CUDA acceleration when available
- **CPU-only:** Falls back to CPU automatically if no GPU detected

Typical performance on M2 Mac:
- **GPU-accelerated:** 34-second video in 10-20 seconds
- **CPU-only:** 40-50 seconds
- **Speedup factor:** 2-3x with current implementation (potential for 3-5x with kernel optimization)

Force CPU-only mode for testing or on systems with GPU issues:
```bash
--device cpu
```

**Window Merging (Jitter Handling):** The sampler merges presence windows separated by ≤ 5 frames to handle card movement jitter during manual placement. This reduces duplicate detections:

- Default merge gap: 5 frames (adjust with `--window-merge-gap` if cards move quickly)
- Example: If a card creates 3 small variance dips due to focus shifts, they merge into 1 continuous window
- Smaller gap (e.g., 2-3): Stricter - only merge closely-spaced windows
- Larger gap (e.g., 8-10): Aggressive - merges windows further apart (may create false positives)


#### When to Use

- Plain, uniform lightbox background (white or neutral color) with colored trading cards
- Manual card placement with brief pause for focus between placements
- No image classification or card detection model available/desired
- Fast processing is important
- Processing on machines without GPU/model installation capability

#### Performance

Contrast-based sampler is approximately **6.4x faster** than ML-based card detection on typical hardware (M2 Mac). No model download or GPU required.

## Sampler Options

The `--sampler` flag selects the frame selection algorithm:

- `stability` (default): Motion-based detection. Scans for stable periods when camera/lighting is steady.
- `detection`: ML-based card detection. Uses YOLO detector to identify card presence.
- `contrast`: Color-variance-based detection. Fast, ML-free detection ideal for controlled lightbox environments.
- `raw`: Simple cadence-based sampling. Samples every N frames without detection logic.

## Advanced Usage

### Review Extracted Frames

Start the local web UI to review and rate extracted cards:

```bash
card-capture review --db card_capture_output/cards.sqlite --port 8000
```

Then open `http://localhost:8000` in your browser.

### Common Configuration Scenarios

**High-volume production lightbox:**
```bash
card-capture process video.mov \
  --sampler contrast \
  --contrast-threshold 800.0 \
  --min-presence-frames 2 \
  --candidates-per-window 3 \
  --scan-fps 15.0
```
Faster scanning with lower threshold for automated card feed.

**Manual lightbox with careful focus:**
```bash
card-capture process video.mov \
  --sampler contrast \
  --contrast-threshold 1500.0 \
  --min-presence-frames 5 \
  --candidates-per-window 8 \
  --scan-fps 5.0
```
More selective, higher quality candidates, slower processing.

**Mixed lighting/detection preference:**
```bash
card-capture process video.mov \
  --sampler detection \
  --detection-scan-fps 3.0 \
  --min-detection-frames 3 \
  --candidates-per-window 5
```
Use ML-based detection when lighting is variable or background changes.

## Development

Run tests:

```bash
pip install -e ".[test]"
pytest tests/
```

## License

See LICENSE for details.
