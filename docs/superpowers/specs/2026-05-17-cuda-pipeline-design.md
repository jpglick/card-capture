# CUDA Pipeline — Sub-project B Design

**Date:** 2026-05-17
**Builds on:** Sub-project A (vast.ai integration layer)

---

## Overview

The existing MPS pipeline on the Mac Mini uses `AdaptivePresenceSampler` (two-pass presence scan at 192px, valley splits, presence classifier) to select ~90 frames from a 37-second 60fps video before running YOLO. On a 4090 with NVDEC, the scan is the bottleneck and its purpose — filtering empty workspace — is unnecessary when YOLO is fast enough to process every frame.

Sub-project B adds a CUDA execution path that:
- Replaces the presence scan with uniform stride sampling via decord NVDEC
- Runs YOLO in large batches directly on GPU
- Hard-fails if GPU is unavailable (no silent CPU fallback except in explicit dev mode)
- Reuses all downstream steps (track, refine, score, resolve) unchanged — `device="auto"` already resolves to CUDA on a Linux instance with an NVIDIA GPU

The CUDA path is activated by setting `detector = "cuda"` in `card_capture_config.json`. The vastai_worker.py writes this config on instance startup.

---

## What Changes

| Component | Change |
|---|---|
| `src/card_capture/sampler/cuda_sampler.py` | **New** — NVDEC stride sampler |
| `pipeline/steps/detect.py` | **Modify** — new `elif ctx.detector == "cuda"` branch + `_run_cuda_inference()` |
| `src/card_capture/detectors.py` | **Modify** — add `detect_batch(images)` to `CardcaptorUltralyticsDetector` |
| `src/card_capture/config.py` | **Modify** — add `cuda_stride`, `cuda_batch_size` |
| `pipeline/steps/start.py` | **Modify** — add same two fields to RunContext |
| `app/vastai_worker.py` | **Modify** — write cuda config on startup |
| `app/web/src/routes/runs/[run_id]/+page.svelte` | **Modify** — cloud badge |

**Zero changes** to: track, refine, score, resolve, dedup, store, fuse steps. They are already CUDA-ready via `device="auto"`.

---

## Section 1: `CudaSampler`

**File:** `src/card_capture/sampler/cuda_sampler.py`

### Interface

Implements the same iterator protocol as `AdaptivePresenceSampler`: yields `FrameSample` objects, exposes `last_source_fps` and `last_selected_frame_count` attributes for telemetry parity.

```python
class CudaSampler:
    def __init__(
        self,
        video_path: Path,
        stride: int = 2,
        opening_scan_s: float = 2.0,
    ) -> None: ...

    def sample(self, video_path: Path = None, sample_fps: float = None) -> Iterator[FrameSample]: ...
```

### GPU-or-die policy

On construction, `CudaSampler` probes NVDEC availability:

```python
allow_fallback = os.environ.get("CC_CUDA_ALLOW_CPU_FALLBACK", "0") == "1"
try:
    self._ctx = decord.gpu(0)
    # Verify GPU context is functional
    decord.bridge.set_bridge("torch")
except Exception:
    if not allow_fallback:
        raise RuntimeError(
            "CudaSampler requires NVDEC (decord GPU context). "
            "Set CC_CUDA_ALLOW_CPU_FALLBACK=1 to allow CPU fallback in dev/test."
        )
    self._ctx = decord.cpu(0)
```

`CC_CUDA_ALLOW_CPU_FALLBACK=1` is only set in test environments. The vastai_worker.py never sets it.

### Sampling logic

```python
vr = decord.VideoReader(str(video_path), ctx=self._ctx)
self.last_source_fps = vr.get_avg_fps() or 30.0
self.last_selected_frame_count = 0

# Opening window: unconditional dense coverage (same as MPS path)
opening_frames = int(self.last_source_fps * self.opening_scan_s)

indices = list(range(0, opening_frames))  # opening: every frame
stride_indices = list(range(opening_frames, len(vr), self.stride))
all_indices = sorted(set(indices + stride_indices))
self.last_selected_frame_count = len(all_indices)

frames = vr.get_batch(all_indices)  # shape: (N, H, W, C) on GPU
for i, idx in enumerate(all_indices):
    frame_np = frames[i].asnumpy()  # transfer to CPU for downstream compat
    ts_ms = int(idx * 1000 / self.last_source_fps)
    yield FrameSample(frame_index=idx, timestamp_ms=ts_ms,
                      image=frame_np, width=frame_np.shape[1], height=frame_np.shape[0])
```

Note: frames are transferred to CPU NumPy for compatibility with the existing detection pipeline. This is a deliberate simplicity choice — the YOLO detector handles its own GPU upload internally, and avoiding two-step GPU↔CPU transfer for every frame adds complexity without meaningful savings given that YOLO batching dominates inference time.

---

## Section 2: CUDA Inference Path in `detect.py`

### `_build_sampler_detector` — new branch

```python
elif ctx.detector == "cuda":
    from card_capture.sampler.cuda_sampler import CudaSampler
    sampler = CudaSampler(
        video_path=Path(ctx.video_path),
        stride=ctx.cuda_stride,
        opening_scan_s=ctx.opening_scan_s,
    )
    detector = CardcaptorUltralyticsDetector(
        confidence_threshold=ctx.corner_confidence,
        detection_width=640,
        device="cuda",  # explicit — hard-fail if CUDA unavailable
    )
```

`device="cuda"` is explicit rather than `"auto"`. On the instance, CUDA must be present; on the Mac, running `detector="cuda"` intentionally fails fast.

### `_run_cuda_inference` — new function

Single-process, no queues, no subprocesses:

```python
def _run_cuda_inference(
    ctx: RunContext,
    sampler: "CudaSampler",
    detector: "CardcaptorUltralyticsDetector",
    output_dir: Path,
    frame_dir: Path,
) -> DetectOutput:
    """Single-process CUDA inference: decode → batch YOLO → DetectOutput."""
    frames = list(sampler.sample())  # yields FrameSample objects

    detection_rows = []
    accepted_frame_presence = []
    batch_size = ctx.cuda_batch_size

    # True batch inference: pass cuda_batch_size images to YOLO in one call.
    # CardcaptorUltralyticsDetector.detect_batch() wraps model([img, img, ...])
    # which Ultralytics handles natively. If detect_batch() doesn't exist yet,
    # it must be added to the detector as part of this sub-project.
    for batch_start in range(0, len(frames), batch_size):
        batch = frames[batch_start:batch_start + batch_size]
        images = [f.image for f in batch]
        batch_results = detector.detect_batch(images)  # returns list, one per frame (None if no detection)

        for frame, result in zip(batch, batch_results):
            accepted_frame_presence.append(
                (frame.frame_index, frame.timestamp_ms, True)
            )
            if result is not None:
                detection_rows.append({
                    "detection_id": len(detection_rows),
                    "frame_index": frame.frame_index,
                    "timestamp_ms": frame.timestamp_ms,
                    "width": frame.width,
                    "height": frame.height,
                    "corners": [(float(p[0]), float(p[1])) for p in result.corners],
                    "confidence": float(result.confidence),
                    "source_frame_path": "",
                    "triage_metrics": {},
                })

    return DetectOutput(
        frame_count=len(frames),
        accepted_frame_count=len(frames),
        accepted_frame_presence=accepted_frame_presence,
        detection_rows=detection_rows,
        sampler_telemetry={
            "sampler_type": "CudaSampler",
            "last_selected_frame_count": sampler.last_selected_frame_count,
            "last_source_fps": sampler.last_source_fps,
            "cuda_stride": ctx.cuda_stride,
        },
        video_id=ctx.video_id,
    )
```

### Routing in `detect.run()`

```python
if ctx.detector == "cuda":
    return _run_cuda_inference(ctx, sampler, detector, output_dir, frame_dir)
else:
    # existing _run_pipeline_workers path — unchanged
    stats, consumer_stats, raw_rows = _run_pipeline_workers(...)
```

---

## Section 3: Config Additions

New fields in `PipelineConfig` (`src/card_capture/config.py`) and `RunContext` (`pipeline/steps/start.py`):

| Field | Default | Purpose |
|---|---|---|
| `cuda_stride: int` | `2` | Sample every Nth source frame (2 = 30fps effective from 60fps) |
| `cuda_batch_size: int` | `32` | YOLO frames per inference batch on the 4090 |

Both wired through `init_run()` the same way as all other config fields.

---

## Section 4: Activation in `vastai_worker.py`

On instance startup, before calling the Metaflow flow, the worker writes CUDA config into `card_capture_config.json`:

```python
_CUDA_CONFIG_OVERRIDES = {
    "detector": "cuda",
    "device": "cuda",
    "cuda_stride": 2,
    "cuda_batch_size": 32,
    "pipeline_backend": "cuda",
}

def _apply_cuda_config() -> dict:
    """Write CUDA overrides; return original values for restore."""
    cfg = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}
    original = {k: cfg.get(k) for k in _CUDA_CONFIG_OVERRIDES}
    cfg.update(_CUDA_CONFIG_OVERRIDES)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return original

def _restore_config(original: dict) -> None:
    cfg = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}
    cfg.update(original)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
```

Called in `_run_pipeline` as:
```python
original = _apply_cuda_config()
try:
    subprocess.run([..., "--detector", "cuda", ...], ...)
finally:
    _restore_config(original)
```

---

## Section 5: Cloud Badge on Runs Page

In `app/web/src/routes/runs/[run_id]/+page.svelte`, the run header gets a small badge when the run was executed on the cloud GPU:

```svelte
{#if run.detector === 'cuda' || run.run_id?.startsWith('batch_')}
    <span class="cloud-badge">☁ Cloud GPU</span>
{/if}
```

`RunDetail` already includes config metadata. No new API endpoint needed.

---

## Testing

**`CudaSampler` unit tests** (with `CC_CUDA_ALLOW_CPU_FALLBACK=1`):
- Stride=2 on a 10-frame synthetic video yields frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for opening window + [every 2nd] for remainder (deduped)
- `last_selected_frame_count` is set correctly
- Raises `RuntimeError` when GPU unavailable and fallback not allowed

**`detect_batch()` unit test** (mocked YOLO model):
- Passes all images in one list to the underlying model call
- Returns a list of the same length as input (None entries for non-detections)

**`_run_cuda_inference` unit tests** (mocked CudaSampler + mocked detector):
- Returns `DetectOutput` with correct `frame_count` and `accepted_frame_count`
- Batches correctly: 70 frames with batch_size=32 → 3 `detect_batch()` calls
- No detections → empty `detection_rows`, non-empty `accepted_frame_presence`

**Integration**: The existing Metaflow tests are unaffected since `detector="cuda"` is not the default.
