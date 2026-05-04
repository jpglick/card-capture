# Kornia GPU Refinement & Lazy Warping Plan

**Goal:** Eliminate redundant image processing by (1) delaying rectification until canonical selection and (2) accelerating those final warps using Kornia on the GPU.

## 1. Architectural Changes
- **Raw Tracking:** Store only `source_frame_path` and `corners` for candidates during tracking.
- **Canonical Selection:** Select best frames using score and sharpness based on metadata; do not warp until selected.
- **Kornia Accelerator:** Create a `KorniaNormalizer` that runs only on selected canonical candidates for the final batch.

---

## 2. Implementation Tasks

### Task 1: Pipeline Refactoring (`src/card_capture/pipeline.py`)
- [ ] Modify processing loop to append raw data `(row.source_frame_path, candidate.corners)` to `track.candidates` instead of pre-warping candidates.
- [ ] In `pipeline.py`, after `_select_canonical_entries`, compile a batch of these raw `(path, corners)` pairs for the session.

### Task 2: GPU Refinement (`src/card_capture/gpu_refinement.py`)
- [ ] Implement `KorniaNormalizer.warp_canonical_batch(list_of_tuples)`:
    - Load images from paths.
    - Move to `mps` device.
    - Perform `kornia` warp/resize.
    - Return normalized tensors/arrays.

### Task 3: Performance Validation
- [ ] Monitor `t_refine` telemetry—it should now represent only a single batch warp per session rather than a continuous cost.

---
## 3. Validation
1. **Performance:** Telemetry data should show a massive drop in total processing time, as normalization cost becomes constant per session rather than linear per frame.
2. **Memory:** Verify the batch size for `warp_canonical_batch` does not spike VRAM.
