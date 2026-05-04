# Card Capture Pipeline: High-Fidelity Refactor Plan

**Goal:** Permanently resolve micro-fragmentation and deduplication failures by implementing spatial track patience, derivative flip tracking, and center-crop hashing.

## 1. Track Fragmentation Fixes
- **Increase Finalization Threshold:** Increase `min_track_length` in `HysteresisTracker` from 3 to 12.
- **Implement Track Coasting:** In `HysteresisTracker`, when `null_detector` returns empty, do not trigger a reset immediately. Coast tracks for 15 frames. If a new detection occurs within the spatial vicinity, merge the segments.
- **Derivative Flip Tracking:** In `detect_flip`, replace the static 20% area floor with a check for a continuous area drop of 30-40% over 5 frames, followed by a detection dropout.

## 2. Deduplication & Hashing Fixes
- **Center-Crop Hashing:** In `VisualDeduplicator.compute_phash`, apply a 20% center-crop to the input image before hashing, ensuring the borders and high-glare edges are excluded.
- **Strict Finalization:** Tracks with less than 12 frames are to be discarded (pruned) immediately during `finalize()`.

---

## 3. Implementation Tasks

### Task 1: Refine Tracking & Gating (`src/card_capture/selector.py`)
- [ ] Modify `HysteresisTracker.__init__` to accept `min_track_length=12` and implement the `max_gap_frames` (coasting) logic.
- [ ] Implement the derivative area drop logic in `detect_flip`.

### Task 2: Refine Deduplication (`src/card_capture/deduplicator.py`)
- [ ] Update `VisualDeduplicator.compute_phash` to perform a 20% center-crop on the input image.

### Task 3: Configuration & Pipeline (`src/card_capture/pipeline.py`)
- [ ] Update `PipelineConfig` and `ProcessingOptions` to include new tracking parameters.
- [ ] Modify `VideoProcessor` to apply the track coasting logic in the processing loop.

### Validation Plan
1. **Micro-track Check:** Query the database after processing the test video; track count should drop from >50 to ~15-20.
2. **Visual Similarity:** Check the `visual_hash` values for deduplicated cards to ensure they are significantly closer (Hamming distance <= 2).
3. **Ghost Suppression:** Verify that the 12-frame min track length successfully prunes the 3-frame noise tracks.
