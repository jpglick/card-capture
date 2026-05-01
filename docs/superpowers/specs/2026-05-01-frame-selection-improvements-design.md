# Frame Selection Improvements — Design Spec

**Date:** 2026-05-01  
**Status:** Approved

---

## Problem

The current pipeline samples frames at a fixed cadence and runs YOLO detection on every sampled frame. This produces poor candidate images because:

1. Mid-flip frames (foreshortened card) are scored alongside stable frames and can win.
2. Every frame is decoded even though most carry no useful new information.
3. The quality score has no concept of card aspect ratio or image complexity, so a featureless card back can score as well as a front.
4. Detection runs at full resolution even though YOLO internally resizes to 640 px — pure overhead on Apple Silicon.

The card is always shown face-up first. The goal is a single best-quality still of the front face per video.

---

## Approach

Four targeted changes to the pipeline, applied in order of where they take effect:

1. Downscale frames before detection (faster inference, same crop quality)
2. Replace cadence sampling with two-pass stability-based sampling (detect still windows cheaply, run YOLO only there)
3. Extend the quality score to penalise mid-flip aspect ratios and featureless backs
4. Early stop after the first stable window yields a good detection

---

## Design

### 1. Detection downscaling

**Where:** `CardcaptorUltralyticsDetector.detect()`

Before calling YOLO, downscale the frame to `detection_width` (default `640`) using `cv2.resize` with proportional scaling: `target_height = round(original_height * detection_width / original_width)`. YOLO already resizes internally to 640 — passing a full-resolution frame is wasted memory and compute on M-series hardware. If `original_width <= detection_width`, skip the resize and set `scale = 1.0` to avoid upscaling low-resolution inputs.

After inference, scale polygon coordinates back to original-resolution space using separate x and y scale factors: `scale_x = original_width / scaled_width`, `scale_y = original_height / scaled_height`. Using separate factors avoids sub-pixel drift caused by rounding `target_height`. The cropper always operates on the original full-resolution frame.

**New parameter:** `detection_width: int = 640` on `CardcaptorUltralyticsDetector`.  
**CLI flag:** `--detection-width` (default 640; 320 for faster, 1280 for harder-to-detect cards).

No changes to `CardCropper` or downstream code.

---

### 2. Two-pass `StabilityBasedSampler`

Replaces `VideoSampler` as the production sampler. Eliminates the need for in-pipeline motion gating. `VideoSampler` stays available as a `--sampler raw` fallback.

#### Pass 1 — stability scan

Sequential decode at `scan_fps` (default `10`) with frames downscaled to `scan_width` (default `160`) wide. For each consecutive pair of downscaled grayscale frames, compute mean absolute pixel difference. Track runs of stable frames where diff < `motion_threshold` (default `8.0` on 0–255 scale). A run must be at least `min_stable_frames` (default `5`) consecutive frames to qualify as a stable window.

During pass 1, for each decoded frame the scan tracks both the **source video frame number** (the value that would be passed to `cv2.CAP_PROP_POS_FRAMES` to seek back to that frame) and the Laplacian variance of the downscaled grayscale image. The source frame number is not the scan-pass counter — at 10 fps scan of a 30 fps source, the actual frame numbers are 0, 3, 6, … and those are the values stored. At the end of each stable window, the source frame number with the highest Laplacian variance is recorded as `StableWindow.best_frame_index`. No raw pixel data is retained after each frame is processed, keeping memory O(1).

Output: list of `StableWindow(start_frame, end_frame, best_frame_index)`. `start_frame` and `end_frame` are retained for logging and debugging (e.g., the review UI could show which time range a card was captured from); they are not used by Pass 2. `StableWindow` is a private `dataclass` defined inside `sampler.py` — it is not added to `models.py` and is not part of the public API. Tests that need the type import it directly from `card_capture.sampler`.

Pass 1 does not call the detector. It is cheap: 160 px frames at 10 fps for a 30s clip is ~300 tiny images.

**No stable windows:** If pass 1 finds zero qualifying windows (e.g., constant camera motion or very short clip), `StabilityBasedSampler.sample()` yields nothing — an empty iterator. The pipeline already handles zero detections gracefully (status `"no_detections"`). No fallback to cadence sampling occurs; use `--sampler raw` explicitly if cadence is preferred.

#### Pass 2 — targeted yield

For each stable window (in temporal order):

1. Seek the video capture to `window.best_frame_index` using `cv2.CAP_PROP_POS_FRAMES`.
2. Decode one frame with `capture.read()`. Use `window.best_frame_index` as the `FrameSample.frame_index` value (do **not** re-read `CAP_PROP_POS_FRAMES` after the read — OpenCV advances it to the next frame on read, which would be off by one). Read `CAP_PROP_POS_MSEC` *before* `capture.read()` for `timestamp_ms`.
3. Yield the full-resolution `FrameSample`.

For a 30s/30fps video with one stable window, this is 1–3 YOLO calls.

Both pass 1 and pass 2 open separate `cv2.VideoCapture` handles and release them in `try/finally` blocks, matching the resource-cleanup convention used by `VideoSampler`. When the pipeline's early-stop breaks out of the generator, Python's `GeneratorExit` mechanism triggers the `finally` and releases the handle — no explicit `sampler.close()` call is needed.

#### Interface compatibility

`StabilityBasedSampler.sample(video_path, sample_fps)` keeps the same signature as `VideoSampler`. The `sample_fps` argument is **ignored** — `scan_fps` is set only via the constructor. This is intentional: the two-pass design doesn't have a meaningful "output FPS" concept. The argument exists solely for interface compatibility with existing test harnesses that call `.sample(path, fps)`.

#### New parameters (exposed via CLI)

| Parameter | Default | CLI flag | Meaning |
|---|---|---|---|
| `scan_fps` | `10` | `--scan-fps` | Frames-per-second for pass 1 diff scan |
| `scan_width` | `160` | `--scan-width` | Width (px) for downscaled diff scan |
| `motion_threshold` | `8.0` | `--motion-threshold` | Max mean pixel diff to count as stable |
| `min_stable_frames` | `5` | `--min-stable-frames` | Min consecutive stable frames for a window |

**`--sampler` flag values:** `stability` (default, uses `StabilityBasedSampler`) and `raw` (uses `VideoSampler`). When `--detector fake` is specified, the sampler is always `SyntheticSampler` regardless of `--sampler`.

---

### 3. Quality score additions

**Where:** `QualityScorer.score(image, detection_confidence)`

Two new components added. Weights rebalanced to sum to 1.0:

| Component | Weight | Measurement |
|---|---|---|
| sharpness | 0.30 | Laplacian variance / 1000, clamped to [0, 1] |
| glare | 0.20 | 1 – (overexposed fraction × 4), clamped to [0, 1] |
| aspect_ratio | 0.20 | Proximity to standard card ratio 0.714 (63.5 mm / 88.9 mm) |
| size | 0.15 | Crop pixel area / target_pixels, clamped to [0, 1] |
| complexity | 0.10 | Std-dev of pixel values normalised to [0, 1]; fronts (artwork) score higher than plain backs |
| confidence | 0.05 | Detector confidence, clamped to [0, 1] |

**Aspect ratio scoring:** `score = clamp(1.0 – abs(actual_ratio – CARD_RATIO) / TOLERANCE)` where `actual_ratio = crop_width / crop_height`, `CARD_RATIO = 0.714`, and `TOLERANCE = 0.25`. A foreshortened mid-flip card (e.g., ratio 0.3) scores ≈ 0.

**Complexity scoring:** Grayscale std-dev (same grayscale conversion as sharpness: `cv2.cvtColor(BGR, COLOR_BGR2GRAY)` or no-op if already gray) divided by `80.0`, clamped to [0, 1]. Value 80 is a rough empirical midpoint; card fronts with art easily reach std-dev > 40, plain backs are typically < 20. This is not guaranteed discrimination but adds signal.

No changes to `QualityScore` model — the new components are added to the `components` dict under keys `"aspect_ratio"` and `"complexity"`.

---

### 4. Early stop

**Where:** `VideoProcessor.process()` pass-2 loop.

New `ProcessingOptions` field: `detections_to_stop: int = 1` (default 1 — stop after the first stable window that yields a successful detection). If the detection passes the `confidence_threshold` and quality score exceeds `quality_floor` (default `0.5`), processing halts immediately without seeking to further stable windows. Set to `0` to disable.

Because the card face is always shown first, stopping after the first good detection is correct in the common case. If no detection passes the floor, processing continues through all stable windows.

**CLI flags:** `--detections-to-stop` (default 1), `--quality-floor` (default 0.5).

---

## What Changes and What Doesn't

| File | Change |
|---|---|
| `sampler.py` | Add `StabilityBasedSampler`; keep `VideoSampler` |
| `detectors.py` | Add downscale + rescale in `CardcaptorUltralyticsDetector.detect()` |
| `scoring.py` | Add `aspect_ratio` and `complexity` components; rebalance weights |
| `pipeline.py` | Add `detections_to_stop`, `quality_floor` to `ProcessingOptions`; add early-stop logic |
| `cli.py` | Add `--detection-width`, `--scan-fps`, `--scan-width`, `--motion-threshold`, `--min-stable-frames`, `--sampler` (`stability`\|`raw`), `--detections-to-stop`, `--quality-floor` flags |
| `models.py` | No changes |
| `cropper.py` | No changes |
| `selector.py` | No changes |
| `storage.py` | No changes |

---

## Testing

- `test_sampler.py` (new): stable window detection with correct source frame numbers in `best_frame_index`, single-best-frame selection per window, no-stable-windows yields empty iterator, fallback to `VideoSampler` with `--sampler raw`
- `test_scoring_selector.py`: add cases for aspect_ratio (mid-flip penalty) and complexity (back vs front)
- `test_pipeline.py`: add `StabilityBasedSampler` smoke test; verify early stop halts after first good detection
- Existing tests remain unchanged and continue passing

---

## Non-Goals

- GPU / CoreML acceleration (separate concern)
- Multi-card videos
- Detecting which face (front vs back) by reading card artwork — the aspect ratio + complexity heuristic is sufficient
- Optical flow flip-point segmentation
