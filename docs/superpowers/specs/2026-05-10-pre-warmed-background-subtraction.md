# Design: Pre-Warmed Background Subtraction

**Date:** 2026-05-10  
**Status:** Approved  
**Context:** Pipeline V4 Phase C (Fixing Card Stand Phantoms)

---

## Problem
The BoT-SORT backend is robust enough to link static background objects (the card stand) into stable tracks, creating "phantom" card detections. The existing `NullStateDetector` is designed to solve this by filtering out static pixels, but it is currently:
1. **Unwired:** It is instantiated in the pipeline worker but never actually called to filter frames.
2. **Un-warmed:** It requires a 30-frame warmup from the *start* of the video, which fails if a card is already present in those frames.

---

## Approach: Global Minimal Presence (Approach A)
We will use the existing Fast Scan pass to identify the globally "least card-like" frames to build a robust background model, even if the video starts with a card.

### 1. Global Background Discovery (`sampler/__init__.py`)
During the 15fps Fast Scan, the `AdaptivePresenceSampler` will maintain a small buffer (default 5) of frames with the lowest `presence_score`.

*   **Logic:** As each frame is scored in Pass 1, if its score is lower than the highest score in our "background buffer", we replace that buffer entry with the new frame.
*   **Safety Threshold:** If the lowest score in the entire video is > 0.4 (configurable), we conclude no reliable background was found and disable filtering for that run to avoid false negatives.

### 2. Pre-Warming the Model (`pipeline.py`)
Before starting the ML producer/consumer workers:
*   The `VideoProcessor` retrieves the background proxy images from the sampler.
*   The `NullStateDetector` is initialized with these frames immediately, bypassing the 30-frame "live" warmup.

### 3. Active Filtering (`pipeline.py`)
In the `_producer_main` loop:
*   Every accepted frame is passed to `null_detector.is_workspace_empty(frame)`.
*   If `True`, the frame is dropped before reaching the detection queue. This ensures the tracker never sees the stand.

---

## Component Changes

### `NullStateDetector` (`pipeline.py`)
*   Add `warmup_batch(frames: list[np.ndarray])` method to initialize the internal background model from pre-selected images.

### `AdaptivePresenceSampler` (`sampler/__init__.py`)
*   Add `background_proxies: list[np.ndarray]` property.
*   Update `_scan_video` to track the N lowest-scoring frames.

### `VideoProcessor` & `_producer_main` (`pipeline.py`)
*   Wire the `is_workspace_empty` call into the producer loop.
*   Pass the background proxies into the producer process.

---

## Testing Plan
1. **Unit Test (`test_detectors.py`):** Verify `NullStateDetector.warmup_batch` produces a stable model.
2. **Integration Test (`test_pipeline.py`):** Mock a video where the first 10 frames have a "card" but frames 20-30 are "empty". Verify the global scan picks the empty frames and the final output does not contain the card stand.
3. **Regression:** Ensure no sharp card frames are accidentally dropped (false negatives).
