# GPU Video Decode Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CPU sequential video decoding in the detect and refine steps with NVDEC GPU decode, eliminating the two main pipeline bottlenecks (detect: 98s → ~5-10s, refine: 113s → ~1-3s).

**Architecture:** A new `decode_frames_gpu` helper in `pipeline_utils.py` does random-access GPU frame decode. The refine step computes the union of all needed frame indices upfront and calls this helper once, sharing the result dict between the Laplacian scan and the Kornia warp. The `CudaSampler` replaces chunked `get_batch` with `decord.VideoLoader` for continuous streaming to YOLO. No silent CPU fallback anywhere — `CC_CUDA_ALLOW_CPU_FALLBACK=1` is the only opt-in.

**Tech Stack:** decord (VideoLoader, VideoReader, gpu(0)), numpy, existing Kornia/OpenCV pipeline

---

## File Map

| File | Action | What changes |
|---|---|---|
| `src/card_capture/pipeline_utils.py` | Modify | Add `decode_frames_gpu`; add `_compute_laplacian_scan_indices`; update `_laplacian_select_frames` signature |
| `pipeline/steps/refine.py` | Modify | Compute index union upfront; call `decode_frames_gpu` once; pass dict to both consumers |
| `src/card_capture/sampler/cuda_sampler.py` | Modify | Replace `_build_indices` + chunked `get_batch` with `VideoLoader` |
| `tests/test_pipeline_utils_gpu.py` | Create | Tests for `decode_frames_gpu` and `_compute_laplacian_scan_indices` |
| `tests/test_cuda_sampler.py` | Create | Tests for VideoLoader-based `CudaSampler` |

---

## Task 1: Add `decode_frames_gpu` and `_compute_laplacian_scan_indices` to pipeline_utils.py

**Files:**
- Modify: `src/card_capture/pipeline_utils.py`
- Create: `tests/test_pipeline_utils_gpu.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline_utils_gpu.py`:

```python
"""Tests for GPU frame decode helpers in pipeline_utils."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_decode_frames_gpu_returns_index_map(monkeypatch):
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_batch = MagicMock()
    fake_batch.__getitem__ = lambda self, i: MagicMock(asnumpy=lambda: fake_frame)

    fake_vr = MagicMock()
    fake_vr.get_batch.return_value = fake_batch

    fake_decord = MagicMock()
    fake_decord.gpu.return_value = "gpu_ctx"
    fake_decord.VideoReader.return_value = fake_vr

    monkeypatch.setattr("card_capture.pipeline_utils.decord", fake_decord, raising=False)

    from card_capture.pipeline_utils import decode_frames_gpu
    result = decode_frames_gpu("/fake/video.mov", [5, 2, 8])

    # Indices passed to get_batch must be sorted
    fake_vr.get_batch.assert_called_once_with([2, 5, 8])
    assert set(result.keys()) == {2, 5, 8}


def test_decode_frames_gpu_hard_fails_without_flag(monkeypatch):
    import os
    monkeypatch.delenv("CC_CUDA_ALLOW_CPU_FALLBACK", raising=False)

    fake_decord = MagicMock()
    fake_decord.gpu.side_effect = RuntimeError("no GPU")
    monkeypatch.setattr("card_capture.pipeline_utils.decord", fake_decord, raising=False)

    from card_capture.pipeline_utils import decode_frames_gpu
    with pytest.raises(RuntimeError, match="CC_CUDA_ALLOW_CPU_FALLBACK"):
        decode_frames_gpu("/fake/video.mov", [0, 1])


def test_decode_frames_gpu_cpu_fallback_with_flag(monkeypatch):
    monkeypatch.setenv("CC_CUDA_ALLOW_CPU_FALLBACK", "1")

    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_batch = MagicMock()
    fake_batch.__getitem__ = lambda self, i: MagicMock(asnumpy=lambda: fake_frame)
    fake_vr = MagicMock()
    fake_vr.get_batch.return_value = fake_batch

    fake_decord = MagicMock()
    fake_decord.gpu.side_effect = RuntimeError("no GPU")
    fake_decord.cpu.return_value = "cpu_ctx"
    fake_decord.VideoReader.return_value = fake_vr
    monkeypatch.setattr("card_capture.pipeline_utils.decord", fake_decord, raising=False)

    from card_capture.pipeline_utils import decode_frames_gpu
    result = decode_frames_gpu("/fake/video.mov", [3])
    fake_decord.cpu.assert_called_once_with(0)
    assert 3 in result


def test_decode_frames_gpu_empty_indices():
    from card_capture.pipeline_utils import decode_frames_gpu
    result = decode_frames_gpu("/fake/video.mov", [])
    assert result == {}


def test_compute_laplacian_scan_indices_basic():
    from card_capture.pipeline_utils import _compute_laplacian_scan_indices
    track_ranges = [
        {"instance_id": "a", "detections": [(10, []), (20, [])]},
    ]
    result = _compute_laplacian_scan_indices(track_ranges, scan_stride=5)
    # range(10, 21, 5) = {10, 15, 20}
    assert result == {10, 15, 20}


def test_compute_laplacian_scan_indices_multiple_tracks():
    from card_capture.pipeline_utils import _compute_laplacian_scan_indices
    track_ranges = [
        {"instance_id": "a", "detections": [(0, []), (4, [])]},
        {"instance_id": "b", "detections": [(10, []), (12, [])]},
    ]
    result = _compute_laplacian_scan_indices(track_ranges, scan_stride=2)
    assert {0, 2, 4} <= result
    assert {10, 12} <= result


def test_compute_laplacian_scan_indices_empty():
    from card_capture.pipeline_utils import _compute_laplacian_scan_indices
    assert _compute_laplacian_scan_indices([], scan_stride=4) == set()
    assert _compute_laplacian_scan_indices([{"instance_id": "a", "detections": []}], scan_stride=4) == set()
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /Users/josh/code/card-capture && python3 -m pytest tests/test_pipeline_utils_gpu.py -v 2>&1 | head -20
```

Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Add `decode_frames_gpu` and `_compute_laplacian_scan_indices` to `pipeline_utils.py`**

Add at the bottom of `src/card_capture/pipeline_utils.py` (before any existing `_laplacian_select_frames` definition if there are two, add after the last top-level function):

```python
# ---------------------------------------------------------------------------
# GPU frame decode
# ---------------------------------------------------------------------------

try:
    import decord as decord  # noqa: F401 — imported for type checking; real import in functions
except ImportError:
    decord = None  # type: ignore[assignment]


def decode_frames_gpu(video_path, indices: list) -> dict:
    """Decode specific frames via NVDEC; return {frame_index: np.ndarray}.

    Raises RuntimeError if GPU context unavailable and
    CC_CUDA_ALLOW_CPU_FALLBACK is not set.
    """
    import os
    if not indices:
        return {}

    import decord as _decord
    try:
        ctx = _decord.gpu(0)
    except Exception:
        if os.environ.get("CC_CUDA_ALLOW_CPU_FALLBACK", "0") == "1":
            ctx = _decord.cpu(0)
        else:
            raise RuntimeError(
                "decode_frames_gpu requires NVDEC (decord GPU context). "
                "Set CC_CUDA_ALLOW_CPU_FALLBACK=1 to allow CPU fallback "
                "in dev/test environments."
            )

    sorted_indices = sorted(set(indices))
    vr = _decord.VideoReader(str(video_path), ctx=ctx)
    frames = vr.get_batch(sorted_indices)
    result = {idx: frames[i].asnumpy() for i, idx in enumerate(sorted_indices)}
    del frames
    return result


def _compute_laplacian_scan_indices(track_ranges: list, scan_stride: int) -> set:
    """Return the set of frame indices that _laplacian_select_frames will scan."""
    all_scan_frames: set = set()
    for t in track_ranges:
        dets = sorted(t.get("detections", []), key=lambda x: x[0])
        if not dets:
            continue
        first_frame, last_frame = dets[0][0], dets[-1][0]
        all_scan_frames |= set(range(first_frame, last_frame + 1, scan_stride))
    return all_scan_frames
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/josh/code/card-capture && python3 -m pytest tests/test_pipeline_utils_gpu.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline_utils.py tests/test_pipeline_utils_gpu.py
git commit -m "feat(perf): add decode_frames_gpu and _compute_laplacian_scan_indices"
```

---

## Task 2: Update `_laplacian_select_frames` to accept pre-decoded frames

**Files:**
- Modify: `src/card_capture/pipeline_utils.py`

The function currently does a VideoCapture sequential decode internally. Add a `decoded_frames` parameter — when provided, skip VideoCapture and look up frames from the dict directly.

- [ ] **Step 1: Add test for `decoded_frames` parameter**

Append to `tests/test_pipeline_utils_gpu.py`:

```python
def test_laplacian_select_frames_uses_decoded_dict(tmp_path, monkeypatch):
    """When decoded_frames is provided, VideoCapture must NOT be opened."""
    import numpy as np
    from card_capture.pipeline_utils import _laplacian_select_frames

    # Sharp frame at index 5, blurry at index 7
    sharp = np.ones((100, 100, 3), dtype=np.uint8) * 128
    sharp[40:60, 40:60] = 255  # high-frequency edge → high Laplacian
    blurry = np.ones((100, 100, 3), dtype=np.uint8) * 128  # uniform → low Laplacian

    decoded = {5: sharp, 6: blurry, 7: blurry}

    track_ranges = [
        {"instance_id": "t1", "detections": [(5, [[0, 0], [1, 0], [1, 1], [0, 1]]),
                                              (7, [[0, 0], [1, 0], [1, 1], [0, 1]])]},
    ]

    # _open_capture must not be called
    monkeypatch.setattr(
        "card_capture.pipeline_utils._open_capture",
        lambda p: (_ for _ in ()).throw(AssertionError("VideoCapture opened unexpectedly")),
    )

    result = _laplacian_select_frames(
        "/fake/video.mov", track_ranges,
        scan_stride=1, top_k=1, max_corner_gap=15,
        decoded_frames=decoded,
    )
    assert "t1" in result
    assert len(result["t1"]) == 1
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd /Users/josh/code/card-capture && python3 -m pytest tests/test_pipeline_utils_gpu.py::test_laplacian_select_frames_uses_decoded_dict -v
```

Expected: FAIL (function signature doesn't accept `decoded_frames`)

- [ ] **Step 3: Update `_laplacian_select_frames` signature and body**

Find the `_laplacian_select_frames` function in `src/card_capture/pipeline_utils.py`. Change its signature and add the early-exit branch when `decoded_frames` is provided.

Change:
```python
def _laplacian_select_frames(
    video_path,
    track_ranges: list,
    scan_stride: int = 4,
    top_k: int = 1,
    max_corner_gap: int = 15,
) -> dict:
```

To:
```python
def _laplacian_select_frames(
    video_path,
    track_ranges: list,
    scan_stride: int = 4,
    top_k: int = 1,
    max_corner_gap: int = 15,
    decoded_frames: "Optional[dict]" = None,
) -> dict:
```

Then find the VideoCapture block (the `try: capture = _open_capture(...)` section) and wrap it with a conditional:

Replace:
```python
    # Single forward video pass — compute Laplacian for every scan frame
    max_scan_frame = max(all_scan_frames)
    try:
        capture = _open_capture(_Path(video_path))
    except Exception:
        return {}

    try:
        curr = 0
        while curr <= max_scan_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if curr in all_scan_frames:
                h, w = frame.shape[:2]
                # Downscale to 640px wide for speed (~1-2ms per frame)
                scale = 640 / w if w > 640 else 1.0
                small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                for ti in track_info.values():
                    if curr in ti["scan_frames"]:
                        ti["scores"][curr] = lap_var
            curr += 1
    finally:
        capture.release()
```

With:
```python
    # Compute Laplacian for every scan frame
    if decoded_frames is not None:
        # Fast path: frames already decoded — no VideoCapture needed
        for curr in all_scan_frames:
            frame = decoded_frames.get(curr)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            scale = 640 / w if w > 640 else 1.0
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            for ti in track_info.values():
                if curr in ti["scan_frames"]:
                    ti["scores"][curr] = lap_var
    else:
        # Slow path: sequential CPU decode via VideoCapture (non-CUDA fallback)
        max_scan_frame = max(all_scan_frames)
        try:
            capture = _open_capture(_Path(video_path))
        except Exception:
            return {}

        try:
            curr = 0
            while curr <= max_scan_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if curr in all_scan_frames:
                    h, w = frame.shape[:2]
                    scale = 640 / w if w > 640 else 1.0
                    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    for ti in track_info.values():
                        if curr in ti["scan_frames"]:
                            ti["scores"][curr] = lap_var
                curr += 1
        finally:
            capture.release()
```

- [ ] **Step 4: Run all pipeline_utils tests**

```bash
cd /Users/josh/code/card-capture && python3 -m pytest tests/test_pipeline_utils_gpu.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline_utils.py tests/test_pipeline_utils_gpu.py
git commit -m "feat(perf): _laplacian_select_frames accepts pre-decoded frames dict — skips VideoCapture"
```

---

## Task 3: Update refine step to decode once and share

**Files:**
- Modify: `pipeline/steps/refine.py`

The refine step currently:
1. Builds `canonical_indices` from track candidates
2. Calls `_laplacian_select_frames` (which opens its own VideoCapture and decodes)
3. Adds Laplacian-selected frames to `canonical_indices`
4. Opens another VideoCapture and decodes `canonical_indices` sequentially

After this task it will:
1. Build `canonical_indices` from track candidates
2. Compute `laplacian_scan_indices` without decoding
3. Call `decode_frames_gpu(union_of_all_indices)` once
4. Pass the decoded dict to `_laplacian_select_frames`
5. Use the same dict for the Kornia warp — no second decode

- [ ] **Step 1: Read the full refine import block and find where to add GPU imports**

The imports at the top of `run()` in `pipeline/steps/refine.py` include:

```python
from card_capture.pipeline_utils import _select_canonical_entries, _glare_mask, _laplacian_heatmap, _compress_array
from card_capture.ingestion import _open_capture
```

- [ ] **Step 2: Update the imports in `run()` to include new helpers**

Find:
```python
    from card_capture.pipeline_utils import _select_canonical_entries, _glare_mask, _laplacian_heatmap, _compress_array
```

Replace with:
```python
    from card_capture.pipeline_utils import (
        _select_canonical_entries, _glare_mask, _laplacian_heatmap, _compress_array,
        decode_frames_gpu, _compute_laplacian_scan_indices,
    )
```

- [ ] **Step 3: Compute index union and call `decode_frames_gpu` once**

Find the section in `run()` that builds `canonical_indices` and calls `_laplacian_select_frames`:

```python
    decoded_images: Dict[int, np.ndarray] = {}
    if canonical_indices:
        sampler_telemetry = track_out.sampler_telemetry
        capture = _open_capture(video_path)
        try:
            curr_idx = 0
            max_target = max(canonical_indices)
            while curr_idx <= max_target:
                ok, frame = capture.read()
                if not ok:
                    break
                if curr_idx in canonical_indices:
                    decoded_images[curr_idx] = frame
                curr_idx += 1
        finally:
            capture.release()
```

Replace that entire block with:

```python
    # Compute union of all frame indices needed by both Laplacian scan and Kornia warp,
    # then decode once via NVDEC instead of two separate CPU VideoCapture passes.
    _lap_scan_indices = _compute_laplacian_scan_indices(_lap_ranges, ctx.laplacian_scan_stride)
    _all_needed = canonical_indices | _lap_scan_indices
    decoded_images: Dict[int, np.ndarray] = {}
    if _all_needed:
        decoded_images = decode_frames_gpu(video_path, sorted(_all_needed))
```

- [ ] **Step 4: Pass `decoded_frames` to `_laplacian_select_frames`**

Find:
```python
        _lap_results = _laplacian_select_frames(
            video_path,
            _lap_ranges,
            scan_stride=ctx.laplacian_scan_stride,
            top_k=_lap_top_k,
            max_corner_gap=ctx.max_corner_gap_frames,
        )
```

Replace with:
```python
        _lap_results = _laplacian_select_frames(
            video_path,
            _lap_ranges,
            scan_stride=ctx.laplacian_scan_stride,
            top_k=_lap_top_k,
            max_corner_gap=ctx.max_corner_gap_frames,
            decoded_frames=decoded_images if decoded_images else None,
        )
```

- [ ] **Step 5: Verify `_open_capture` import is no longer needed in refine**

Check if `_open_capture` is used anywhere else in `refine.py`:

```bash
grep -n "_open_capture" pipeline/steps/refine.py
```

If it only appeared in the block we replaced, remove it from the import line:

```python
    from card_capture.ingestion import _open_capture
```

→ Delete this line if `_open_capture` is no longer referenced.

- [ ] **Step 6: Run existing tests to verify nothing broke**

```bash
cd /Users/josh/code/card-capture && python3 -m pytest tests/app/ -q --ignore=tests/app/test_beam_runner.py --ignore=tests/app/test_runpod_runner.py --ignore=tests/app/test_worker_core.py --ignore=tests/app/test_integration.py 2>&1 | tail -5
```

Expected: same pass/fail as before this task

- [ ] **Step 7: Commit**

```bash
git add pipeline/steps/refine.py
git commit -m "feat(perf): refine step decodes all frames once via GPU instead of two CPU VideoCapture passes"
```

---

## Task 4: Replace CudaSampler chunked get_batch with VideoLoader

**Files:**
- Modify: `src/card_capture/sampler/cuda_sampler.py`
- Create: `tests/test_cuda_sampler.py`

VideoLoader handles batching internally. `interval=stride-1` gives uniform stride from frame 0 (opening window removed — unnecessary with full GPU coverage).

- [ ] **Step 1: Write failing tests**

Create `tests/test_cuda_sampler.py`:

```python
"""Tests for CudaSampler VideoLoader-based implementation."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


def _make_batch(n, h=100, w=100):
    """Return a fake VideoLoader (batch_data, batch_indices) pair."""
    data = MagicMock()
    data.asnumpy.return_value = np.zeros((n, h, w, 3), dtype=np.uint8)
    indices = MagicMock()
    indices.asnumpy.return_value = np.arange(n * 2, step=2).reshape(-1)
    return data, indices


def _make_probe_vr(total=60, fps=30.0, h=100, w=100):
    first_frame = MagicMock()
    first_frame.asnumpy.return_value = np.zeros((h, w, 3), dtype=np.uint8)
    vr = MagicMock()
    vr.__len__ = lambda self: total
    vr.get_avg_fps.return_value = fps
    vr.__getitem__ = lambda self, i: first_frame
    return vr


@patch("card_capture.sampler.cuda_sampler.decord")
def test_sample_batches_uses_video_loader(mock_decord):
    mock_decord.gpu.return_value = "gpu_ctx"
    mock_decord.cpu.return_value = "cpu_ctx"

    probe_vr = _make_probe_vr(total=60, fps=30.0, h=100, w=100)
    mock_decord.VideoReader.return_value = probe_vr

    batch1 = _make_batch(4)
    mock_vl = MagicMock()
    mock_vl.__iter__ = MagicMock(return_value=iter([batch1]))
    mock_decord.VideoLoader.return_value = mock_vl

    from card_capture.sampler.cuda_sampler import CudaSampler
    sampler = CudaSampler(stride=2)
    batches = list(sampler.sample_batches(batch_size=4, video_path="/fake/v.mov"))

    # VideoLoader must be constructed with interval=1 (stride-1) for stride=2
    mock_decord.VideoLoader.assert_called_once()
    call_kwargs = mock_decord.VideoLoader.call_args
    assert call_kwargs[1]["interval"] == 1 or call_kwargs[0][3] == 1

    assert len(batches) == 1
    assert len(batches[0]) == 4
    from card_capture.models import FrameSample
    assert isinstance(batches[0][0], FrameSample)


@patch("card_capture.sampler.cuda_sampler.decord")
def test_sample_batches_hard_fails_without_gpu(mock_decord, monkeypatch):
    monkeypatch.delenv("CC_CUDA_ALLOW_CPU_FALLBACK", raising=False)
    mock_decord.gpu.side_effect = RuntimeError("no GPU")

    from card_capture.sampler.cuda_sampler import CudaSampler
    with pytest.raises(RuntimeError, match="CC_CUDA_ALLOW_CPU_FALLBACK"):
        CudaSampler(stride=2)


@patch("card_capture.sampler.cuda_sampler.decord")
def test_sample_yields_frame_samples(mock_decord):
    mock_decord.gpu.return_value = "gpu_ctx"
    mock_decord.cpu.return_value = "cpu_ctx"

    probe_vr = _make_probe_vr(total=4, fps=30.0, h=50, w=80)
    mock_decord.VideoReader.return_value = probe_vr

    batch1 = _make_batch(2, h=50, w=80)
    mock_vl = MagicMock()
    mock_vl.__iter__ = MagicMock(return_value=iter([batch1]))
    mock_decord.VideoLoader.return_value = mock_vl

    from card_capture.sampler.cuda_sampler import CudaSampler
    from card_capture.models import FrameSample
    sampler = CudaSampler(stride=2)
    frames = list(sampler.sample(video_path="/fake/v.mov"))

    assert all(isinstance(f, FrameSample) for f in frames)
    assert all(f.width == 80 and f.height == 50 for f in frames)


@patch("card_capture.sampler.cuda_sampler.decord")
def test_stride_1_uses_interval_0(mock_decord):
    mock_decord.gpu.return_value = "gpu_ctx"
    probe_vr = _make_probe_vr(total=10, fps=30.0)
    mock_decord.VideoReader.return_value = probe_vr
    mock_vl = MagicMock()
    mock_vl.__iter__ = MagicMock(return_value=iter([]))
    mock_decord.VideoLoader.return_value = mock_vl

    from card_capture.sampler.cuda_sampler import CudaSampler
    sampler = CudaSampler(stride=1)
    list(sampler.sample_batches(batch_size=4, video_path="/fake/v.mov"))

    call_kwargs = mock_decord.VideoLoader.call_args
    interval_arg = call_kwargs[1].get("interval", call_kwargs[0][3] if len(call_kwargs[0]) > 3 else None)
    assert interval_arg == 0
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /Users/josh/code/card-capture && python3 -m pytest tests/test_cuda_sampler.py -v 2>&1 | head -20
```

Expected: FAIL (VideoLoader not yet used in CudaSampler)

- [ ] **Step 3: Rewrite `cuda_sampler.py`**

Replace the entire file content:

```python
"""NVDEC-accelerated stride sampler for the CUDA pipeline.

Replaces the AdaptivePresenceSampler on GPU instances.
Uses decord.VideoLoader for continuous batched GPU decode with no OOM risk.

GPU-or-die: raises RuntimeError if NVDEC is unavailable and
CC_CUDA_ALLOW_CPU_FALLBACK is not set. Production containers never set that flag.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np

try:
    import decord
except ImportError:
    decord = None  # type: ignore[assignment]

from card_capture.models import FrameSample


def _probe_gpu() -> object:
    """Return a decord GPU context (index 0), or raise on failure."""
    return decord.gpu(0)


class CudaSampler:
    """Uniform-stride video sampler using decord VideoLoader for GPU decode.

    Args:
        video_path: Source video file.
        stride: Sample every Nth source frame. Default 2 = every other frame.
    """

    def __init__(
        self,
        video_path: Optional[Union[Path, str]] = None,
        stride: int = 2,
        opening_scan_s: float = 2.0,  # retained for API compat, no longer used
    ) -> None:
        self.video_path = Path(video_path) if video_path else None
        self.stride = max(1, stride)
        self.last_source_fps: float = 30.0
        self.last_selected_frame_count: int = 0

        allow_fallback = os.environ.get("CC_CUDA_ALLOW_CPU_FALLBACK", "0") == "1"
        try:
            self._gpu_ctx = _probe_gpu()
        except Exception:
            if not allow_fallback:
                raise RuntimeError(
                    "CudaSampler requires NVDEC (decord GPU context). "
                    "Set CC_CUDA_ALLOW_CPU_FALLBACK=1 to allow CPU fallback "
                    "in dev/test environments."
                )
            self._gpu_ctx = decord.cpu(0)

    def sample(
        self,
        video_path: Optional[Union[Path, str]] = None,
        sample_fps: Optional[float] = None,
    ) -> Iterator[FrameSample]:
        """Yield FrameSample for each selected source frame."""
        for batch in self.sample_batches(batch_size=32, video_path=video_path):
            yield from batch

    def sample_batches(
        self,
        batch_size: int = 32,
        video_path: Optional[Union[Path, str]] = None,
    ) -> Iterator[list]:
        """Yield lists of FrameSample using VideoLoader for continuous GPU streaming.

        VideoLoader handles batch management internally. Only batch_size frames
        are in RAM at once regardless of video length.
        """
        resolved = Path(video_path) if video_path else self.video_path
        if resolved is None:
            raise ValueError("video_path must be provided")

        # Probe video dimensions with a CPU reader (minimal overhead)
        probe = decord.VideoReader(str(resolved), ctx=decord.cpu(0))
        total = len(probe)
        fps = probe.get_avg_fps() or 30.0
        first = probe[0].asnumpy()
        h, w = first.shape[:2]
        self.last_source_fps = fps
        self.last_selected_frame_count = max(1, (total + self.stride - 1) // self.stride)
        del probe

        if total == 0:
            return

        # interval=stride-1: interval=0 → every frame, interval=1 → every 2nd, etc.
        vl = decord.VideoLoader(
            [str(resolved)],
            ctx=[self._gpu_ctx],
            shape=(batch_size, h, w, 3),
            interval=max(0, self.stride - 1),
            skip=0,
            shuffle=0,
        )

        for batch_data, batch_indices in vl:
            frames_np = batch_data.asnumpy()               # (N, H, W, 3)
            indices_flat = batch_indices.asnumpy().reshape(-1).astype(int)

            batch = [
                FrameSample(
                    frame_index=int(idx),
                    timestamp_ms=int(idx * 1000 / fps),
                    image=frames_np[i],
                    width=w,
                    height=h,
                )
                for i, idx in enumerate(indices_flat)
            ]
            yield batch
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/josh/code/card-capture && python3 -m pytest tests/test_cuda_sampler.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/sampler/cuda_sampler.py tests/test_cuda_sampler.py
git commit -m "feat(perf): replace CudaSampler chunked get_batch with VideoLoader — continuous GPU streaming"
```

---

## Task 5: Full verification

- [ ] **Step 1: Run the full new test suite**

```bash
cd /Users/josh/code/card-capture && python3 -m pytest tests/test_pipeline_utils_gpu.py tests/test_cuda_sampler.py -v
```

Expected: all 13 tests PASS

- [ ] **Step 2: Run existing app tests to confirm no regressions**

```bash
cd /Users/josh/code/card-capture && python3 -m pytest tests/app/ -q \
  --ignore=tests/app/test_integration.py \
  --ignore=tests/app/test_beam_runner.py \
  --ignore=tests/app/test_runpod_runner.py \
  --ignore=tests/app/test_worker_core.py \
  2>&1 | tail -5
```

Expected: same pass/fail as before

- [ ] **Step 3: Push**

```bash
git push origin main
```
