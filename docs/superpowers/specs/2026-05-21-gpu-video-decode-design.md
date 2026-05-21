# GPU Video Decode Performance Design

**Date:** 2026-05-21  
**Status:** Approved

## Problem

Two pipeline stages are running far slower than expected on a 4090:

- **detect (98s):** `CudaSampler` loads all selected frames into RAM at once via `vr.get_batch(all_indices)`, then materializes them into a Python list before YOLO runs. For a 30s 4K video this is ~11 GB of simultaneous RAM allocation. Memory pressure dominates; the GPU starves.

- **refine (113s):** Two separate OpenCV CPU sequential decode passes through up to 900 4K frames — one for `_laplacian_select_frames` (Laplacian quality scan) and one for the canonical frame decode (Kornia warp inputs). Both read from frame 0 to `max_target` on CPU just to extract ~16-50 specific frames.

## Goals

- GPU continuously fed during YOLO inference
- Refine decodes each frame at most once, using NVDEC
- No silent CPU fallback anywhere — hard-fail unless `CC_CUDA_ALLOW_CPU_FALLBACK=1`

---

## Change 1: CudaSampler → VideoLoader

**File:** `src/card_capture/sampler/cuda_sampler.py`

Replace the chunked `vr.get_batch` approach with `decord.VideoLoader`.

**Design:**

1. Probe video dimensions with a lightweight CPU `VideoReader` open (one frame read). Store `h`, `w`, `fps`, `total` and close.
2. Construct `VideoLoader([video_path], ctx=[gpu_ctx], shape=(batch_size, h, w, 3), interval=stride-1, skip=0, shuffle=0)`.
3. `sample_batches()` iterates the VideoLoader, converting each `(batch_data, batch_indices)` pair to a list of `FrameSample` objects via `.asnumpy()`.
4. `sample()` delegates to `sample_batches(batch_size=32)` and yields individual `FrameSample` objects.

**Opening window removed:** VideoLoader samples uniformly from frame 0 with `interval=stride-1`. The dense opening window is unnecessary when every Nth frame is examined on GPU — the opening frames are covered by the stride pass.

**Hard-fail:** `_probe_gpu()` raises if `decord.gpu(0)` fails and `CC_CUDA_ALLOW_CPU_FALLBACK` is not set. VideoLoader is constructed with the GPU context only — no CPU fallback path.

**`last_selected_frame_count`:** Set to `ceil(total / stride)` from the probe step, used for telemetry.

---

## Change 2: Refine GPU Decode

**Files:** `src/card_capture/pipeline_utils.py`, `pipeline/steps/refine.py`

### New helper: `decode_frames_gpu`

```python
def decode_frames_gpu(
    video_path: Path,
    indices: list[int],
) -> dict[int, np.ndarray]:
```

- Creates `decord.gpu(0)` context. Raises `RuntimeError` if unavailable and `CC_CUDA_ALLOW_CPU_FALLBACK` is not set; falls back to `decord.cpu(0)` only if flag is set.
- Calls `vr.get_batch(sorted(indices))` — random access to only the needed frames via NVDEC.
- Returns `{frame_index: np.ndarray}`.

### Refine step changes (`pipeline/steps/refine.py`)

1. **Compute index union upfront:** Before any decode, compute `laplacian_indices ∪ canonical_indices` — the union of all frame indices needed by both passes.
2. **Single decode call:** `decoded = decode_frames_gpu(video_path, sorted(union))`.
3. **Pass dict to `_laplacian_select_frames`:** Add a `decoded_frames: dict` parameter. When provided, skip VideoCapture entirely — look up frames from the dict. When not provided (backward-compat), existing OpenCV path still runs. Since refine always provides the dict on CUDA instances, the OpenCV path is unreachable in production but intentionally preserved for non-CUDA local runs (where the calling code skips `decode_frames_gpu` and passes `decoded_frames=None`).
4. **Canonical warp:** Reads from the same `decoded` dict — no second decode pass.

### `_laplacian_select_frames` signature change

```python
def _laplacian_select_frames(
    video_path,
    track_ranges,
    scan_stride,
    top_k,
    max_corner_gap,
    decoded_frames: dict | None = None,  # new
) -> dict:
```

When `decoded_frames` is provided, the VideoCapture loop is skipped. Frame lookup: `decoded_frames.get(curr_frame_index)`.

---

## No CPU Fallback Policy

- `CC_CUDA_ALLOW_CPU_FALLBACK=1` is the single opt-in for CPU fallback across all GPU-accelerated paths
- It is checked in: `CudaSampler._probe_gpu()`, `decode_frames_gpu()`, `probe_torch_device_status()` (already done)
- Production containers never set it
- Local dev/test environments set it explicitly when no GPU is present

---

## Expected Timing After Fix

| Stage | Before | Expected After |
|---|---|---|
| detect | ~98s | ~5-10s |
| refine | ~113s | ~1-3s |
| total pipeline | ~238s | ~30-50s |

---

## Files Changed

| File | Change |
|---|---|
| `src/card_capture/sampler/cuda_sampler.py` | Replace chunked `get_batch` with `VideoLoader` |
| `src/card_capture/pipeline_utils.py` | Add `decode_frames_gpu`; update `_laplacian_select_frames` signature |
| `pipeline/steps/refine.py` | Compute index union, call `decode_frames_gpu` once, pass dict to both consumers |
