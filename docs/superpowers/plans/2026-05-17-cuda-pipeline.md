# CUDA Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `detector="cuda"` path to the existing Metaflow pipeline that replaces the presence scan with NVDEC stride sampling and runs YOLO in large GPU batches, while reusing all downstream steps unchanged.

**Architecture:** `CudaSampler` decodes every Nth source frame via decord NVDEC (hard-fails if GPU absent). `_run_cuda_inference()` in `detect.py` converts frames to `FramePacket` objects and calls the existing `detect_batch()` in batches of `cuda_batch_size`. `_build_sampler_detector()` routes to this path when `ctx.detector == "cuda"`. The vastai_worker writes the cuda config overrides before invoking the flow.

**Tech Stack:** Python, decord (NVDEC), existing `CardcaptorUltralyticsDetector.detect_batch()`, Metaflow, Svelte.

**Spec:** `docs/superpowers/specs/2026-05-17-cuda-pipeline-design.md`

---

## File Map

**New:**
- `src/card_capture/sampler/cuda_sampler.py` — NVDEC stride sampler
- `tests/sampler/test_cuda_sampler.py` — CudaSampler unit tests
- `tests/pipeline/test_detect_cuda.py` — _run_cuda_inference unit tests

**Modified:**
- `src/card_capture/config.py` — add `cuda_stride`, `cuda_batch_size`
- `pipeline/steps/start.py` — add same fields to RunContext + init_run
- `pipeline/steps/detect.py` — new CUDA branch in `_build_sampler_detector` + `_run_cuda_inference()`
- `app/vastai_worker.py` — write/restore cuda config overrides
- `app/web/src/routes/runs/[run_id]/+page.svelte` — cloud GPU badge

---

### Task 1: Add `cuda_stride` and `cuda_batch_size` config fields

**Files:**
- Modify: `src/card_capture/config.py`
- Modify: `pipeline/steps/start.py`

- [ ] **Step 1: Add to `PipelineConfig`**

In `src/card_capture/config.py`, after `cuda_idle_timeout_s: int = 300` (or after the last existing field in the compute group), add:

```python
    cuda_stride: int = 2
    cuda_batch_size: int = 32
```

- [ ] **Step 2: Add to `RunContext` dataclass**

In `pipeline/steps/start.py`, after `cuda_idle_timeout_s: int = 300` in the RunContext dataclass, add:

```python
    cuda_stride: int = 2
    cuda_batch_size: int = 32
```

- [ ] **Step 3: Wire in `init_run`**

In `pipeline/steps/start.py`, after `cuda_idle_timeout_s=cfg.cuda_idle_timeout_s,` in the `RunContext(...)` call inside `init_run`, add:

```python
        cuda_stride=cfg.cuda_stride,
        cuda_batch_size=cfg.cuda_batch_size,
```

- [ ] **Step 4: Run tests to confirm no breakage**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py tests/pipeline/test_score_novelty_gate.py -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/config.py pipeline/steps/start.py
git commit -m "feat(cuda): add cuda_stride and cuda_batch_size config fields"
```

---

### Task 2: Write failing tests for `CudaSampler`

**Files:**
- Create: `tests/sampler/test_cuda_sampler.py`

- [ ] **Step 1: Create the test file**

Create `tests/sampler/test_cuda_sampler.py`:

```python
"""Tests for CudaSampler — uses CC_CUDA_ALLOW_CPU_FALLBACK=1 for GPU-free CI."""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Allow CPU fallback so tests run without a real GPU
os.environ.setdefault("CC_CUDA_ALLOW_CPU_FALLBACK", "1")


def _make_video(tmp_path: Path, n_frames: int = 20, fps: int = 60) -> Path:
    """Write a synthetic video; return its path."""
    path = tmp_path / "test.mp4"
    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64)
    )
    for i in range(n_frames):
        frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


def test_stride_2_yields_correct_indices(tmp_path):
    """20-frame video, stride=2, opening=0 → frames [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]."""
    from card_capture.sampler.cuda_sampler import CudaSampler
    vpath = _make_video(tmp_path, n_frames=20, fps=60)
    sampler = CudaSampler(video_path=vpath, stride=2, opening_scan_s=0.0)
    samples = list(sampler.sample())
    indices = [s.frame_index for s in samples]
    assert indices == list(range(0, 20, 2))


def test_opening_scan_covers_first_seconds(tmp_path):
    """opening_scan_s=0.5 at 60fps → first 30 frames all included, then stride."""
    from card_capture.sampler.cuda_sampler import CudaSampler
    vpath = _make_video(tmp_path, n_frames=60, fps=60)
    sampler = CudaSampler(video_path=vpath, stride=4, opening_scan_s=0.5)
    samples = list(sampler.sample())
    indices = [s.frame_index for s in samples]
    # First 30 frames all present (every frame in opening window)
    for i in range(30):
        assert i in indices, f"Frame {i} missing from opening scan"


def test_frame_samples_have_correct_fields(tmp_path):
    """FrameSample objects have image, width, height, frame_index, timestamp_ms."""
    from card_capture.sampler.cuda_sampler import CudaSampler
    vpath = _make_video(tmp_path, n_frames=5, fps=30)
    sampler = CudaSampler(video_path=vpath, stride=1, opening_scan_s=0.0)
    samples = list(sampler.sample())
    assert len(samples) == 5
    s = samples[0]
    assert s.frame_index == 0
    assert s.image is not None
    assert s.width == 64
    assert s.height == 64
    assert s.timestamp_ms >= 0


def test_last_selected_frame_count_set(tmp_path):
    """last_selected_frame_count is set after sample() is exhausted."""
    from card_capture.sampler.cuda_sampler import CudaSampler
    vpath = _make_video(tmp_path, n_frames=10, fps=60)
    sampler = CudaSampler(video_path=vpath, stride=2, opening_scan_s=0.0)
    list(sampler.sample())
    assert sampler.last_selected_frame_count == 5
    assert sampler.last_source_fps == pytest.approx(60.0, abs=2.0)


def test_raises_without_gpu_when_fallback_not_set(tmp_path, monkeypatch):
    """RuntimeError raised when GPU unavailable and CC_CUDA_ALLOW_CPU_FALLBACK not set."""
    monkeypatch.delenv("CC_CUDA_ALLOW_CPU_FALLBACK", raising=False)
    # Patch decord.gpu to raise so we simulate no-GPU env
    import unittest.mock as mock
    import card_capture.sampler.cuda_sampler as mod
    with mock.patch.object(mod, "_probe_gpu", side_effect=RuntimeError("no GPU")):
        with pytest.raises(RuntimeError, match="NVDEC"):
            vpath = _make_video(tmp_path)
            mod.CudaSampler(video_path=vpath, stride=2, opening_scan_s=0.0)
```

- [ ] **Step 2: Verify tests fail**

```bash
python3 -m pytest tests/sampler/test_cuda_sampler.py -v 2>&1 | tail -10
```

Expected: ImportError — `cannot import name 'CudaSampler'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/sampler/test_cuda_sampler.py
git commit -m "test(cuda): failing tests for CudaSampler"
```

---

### Task 3: Implement `CudaSampler`

**Files:**
- Create: `src/card_capture/sampler/cuda_sampler.py`

- [ ] **Step 1: Create the sampler**

Create `src/card_capture/sampler/cuda_sampler.py`:

```python
"""NVDEC-accelerated stride sampler for the CUDA pipeline.

Replaces the AdaptivePresenceSampler on vast.ai GPU instances.
No presence classifier, no windowing — uniform stride across the full video
with dense coverage for the opening window (catching cards on stands).

GPU-or-die: raises RuntimeError if NVDEC is unavailable and
CC_CUDA_ALLOW_CPU_FALLBACK is not set. The vastai_worker never sets that
variable; it is only for local dev/test environments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np

from card_capture.models import FrameSample


def _probe_gpu() -> object:
    """Return a decord GPU context (index 0), or raise on failure."""
    import decord
    return decord.gpu(0)


class CudaSampler:
    """Uniform-stride video sampler using decord NVDEC for GPU decode.

    Args:
        video_path: Source video file.
        stride: Decode every Nth source frame outside the opening window.
            Default 2 = 30fps effective from a 60fps source.
        opening_scan_s: Always include every frame from the first N seconds,
            regardless of stride, so cards resting on a stand at the start
            are never missed.
    """

    def __init__(
        self,
        video_path: Optional[Union[Path, str]] = None,
        stride: int = 2,
        opening_scan_s: float = 2.0,
    ) -> None:
        self.video_path = Path(video_path) if video_path else None
        self.stride = max(1, stride)
        self.opening_scan_s = max(0.0, opening_scan_s)
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
            import decord
            self._gpu_ctx = decord.cpu(0)

    def sample(
        self,
        video_path: Optional[Union[Path, str]] = None,
        sample_fps: Optional[float] = None,
    ) -> Iterator[FrameSample]:
        """Yield FrameSample for each selected source frame."""
        import decord

        resolved = Path(video_path) if video_path else self.video_path
        if resolved is None:
            raise ValueError("video_path must be provided")

        vr = decord.VideoReader(str(resolved), ctx=self._gpu_ctx)
        total = len(vr)
        self.last_source_fps = vr.get_avg_fps() or 30.0

        # Opening window: every frame (dense — catches cards on stands)
        opening_count = int(self.last_source_fps * self.opening_scan_s)
        opening_indices = list(range(0, min(opening_count, total)))

        # Remaining: every stride-th frame
        stride_indices = list(range(opening_count, total, self.stride))

        # Merge and deduplicate (opening may overlap stride start)
        all_indices = sorted(set(opening_indices + stride_indices))
        self.last_selected_frame_count = len(all_indices)

        if not all_indices:
            return

        # Batch-decode via NVDEC
        frames = vr.get_batch(all_indices)   # shape: (N, H, W, C), GPU or CPU tensor

        for i, idx in enumerate(all_indices):
            frame_np = frames[i].asnumpy()   # transfer to CPU NumPy for downstream compat
            h, w = frame_np.shape[:2]
            ts_ms = int(idx * 1000 / self.last_source_fps)
            yield FrameSample(
                frame_index=idx,
                timestamp_ms=ts_ms,
                image=frame_np,
                width=w,
                height=h,
            )
```

- [ ] **Step 2: Run the 5 new tests**

```bash
python3 -m pytest tests/sampler/test_cuda_sampler.py -v 2>&1 | tail -12
```

Expected: all 5 pass (with `CC_CUDA_ALLOW_CPU_FALLBACK=1` set in the test file).

- [ ] **Step 3: Run broader suite to confirm no regressions**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py tests/sampler/ -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/sampler/cuda_sampler.py tests/sampler/test_cuda_sampler.py
git commit -m "feat(cuda): CudaSampler — NVDEC stride sampler, GPU-or-die"
```

---

### Task 4: Write failing tests for `_run_cuda_inference`

**Files:**
- Create: `tests/pipeline/test_detect_cuda.py`

- [ ] **Step 1: Create failing tests**

Create `tests/pipeline/test_detect_cuda.py`:

```python
"""Tests for _run_cuda_inference — mocked sampler and detector."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

os.environ.setdefault("CC_CUDA_ALLOW_CPU_FALLBACK", "1")


def _make_frame_sample(frame_index: int):
    import numpy as np
    from card_capture.models import FrameSample
    return FrameSample(
        frame_index=frame_index,
        timestamp_ms=frame_index * 16,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        width=64,
        height=64,
    )


def _make_ctx(tmp_path, batch_size=4):
    from pipeline.steps.start import RunContext
    return RunContext(
        video_path=str(tmp_path / "video.mp4"),
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "cards.sqlite"),
        detector="cuda",
        config_preset="balanced",
        cuda_batch_size=batch_size,
        cuda_stride=2,
    )


def test_detect_output_frame_count(tmp_path):
    """DetectOutput.frame_count matches number of sampled frames."""
    from pipeline.steps.detect import _run_cuda_inference
    from card_capture.sampler.cuda_sampler import CudaSampler

    ctx = _make_ctx(tmp_path)
    frames = [_make_frame_sample(i) for i in range(10)]

    sampler = MagicMock(spec=CudaSampler)
    sampler.sample.return_value = iter(frames)
    sampler.last_selected_frame_count = 10
    sampler.last_source_fps = 60.0

    detector = MagicMock()
    detector.confidence_threshold = 0.5
    detector.detect_batch.return_value = []  # no detections

    out = _run_cuda_inference(ctx, sampler, detector, tmp_path, tmp_path)
    assert out.frame_count == 10
    assert out.accepted_frame_count == 10
    assert out.detection_rows == []


def test_detect_output_batching(tmp_path):
    """With batch_size=4 and 10 frames, detect_batch is called 3 times."""
    from pipeline.steps.detect import _run_cuda_inference
    from card_capture.sampler.cuda_sampler import CudaSampler

    ctx = _make_ctx(tmp_path, batch_size=4)
    frames = [_make_frame_sample(i) for i in range(10)]

    sampler = MagicMock(spec=CudaSampler)
    sampler.sample.return_value = iter(frames)
    sampler.last_selected_frame_count = 10
    sampler.last_source_fps = 60.0

    detector = MagicMock()
    detector.confidence_threshold = 0.5
    detector.detect_batch.return_value = []

    _run_cuda_inference(ctx, sampler, detector, tmp_path, tmp_path)
    # 10 frames / batch_size=4 → ceil(10/4) = 3 calls
    assert detector.detect_batch.call_count == 3


def test_detect_output_has_detection_rows(tmp_path):
    """Detections returned by detect_batch appear in detection_rows."""
    from pipeline.steps.detect import _run_cuda_inference
    from card_capture.sampler.cuda_sampler import CudaSampler
    from card_capture.models import DetectionPacket, FramePacket, CornerDetection

    ctx = _make_ctx(tmp_path, batch_size=8)
    frames = [_make_frame_sample(0)]

    sampler = MagicMock(spec=CudaSampler)
    sampler.sample.return_value = iter(frames)
    sampler.last_selected_frame_count = 1
    sampler.last_source_fps = 60.0

    # Fake detection packet
    cd = MagicMock()
    cd.corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    cd.confidence = 0.85
    pkt = MagicMock()
    pkt.frame_index = 0
    pkt.timestamp_ms = 0
    pkt.width = 64
    pkt.height = 64
    pkt.corner_detection = cd

    detector = MagicMock()
    detector.confidence_threshold = 0.5
    detector.detect_batch.return_value = [pkt]

    out = _run_cuda_inference(ctx, sampler, detector, tmp_path, tmp_path)
    assert len(out.detection_rows) == 1
    assert out.detection_rows[0]["confidence"] == pytest.approx(0.85)
    assert out.detection_rows[0]["frame_index"] == 0
```

- [ ] **Step 2: Verify they fail**

```bash
python3 -m pytest tests/pipeline/test_detect_cuda.py -v 2>&1 | tail -10
```

Expected: ImportError — `cannot import name '_run_cuda_inference'` from `pipeline.steps.detect`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/pipeline/test_detect_cuda.py
git commit -m "test(cuda): failing tests for _run_cuda_inference"
```

---

### Task 5: Implement `_run_cuda_inference` and CUDA branch in `detect.py`

**Files:**
- Modify: `pipeline/steps/detect.py`

- [ ] **Step 1: Add `_run_cuda_inference` function**

Read `pipeline/steps/detect.py`. Add this function after `_build_sampler_detector`:

```python
def _run_cuda_inference(
    ctx: RunContext,
    sampler: "CudaSampler",
    detector: "CardcaptorUltralyticsDetector",
    output_dir: Path,
    frame_dir: Path,
) -> "DetectOutput":
    """Single-process CUDA inference path: decode → batch YOLO → DetectOutput.

    Used when ctx.detector == "cuda". Bypasses the multiprocessing
    producer/consumer model — frames are decoded by NVDEC and fed to YOLO
    in large batches without subprocess overhead.
    """
    from card_capture.models import FramePacket

    frames = list(sampler.sample())   # list[FrameSample] — already decoded
    batch_size = ctx.cuda_batch_size

    detection_rows: list[dict] = []
    accepted_frame_presence: list[tuple[int, int, bool]] = []
    det_id = 0

    for batch_start in range(0, len(frames), batch_size):
        batch = frames[batch_start:batch_start + batch_size]

        # Convert FrameSample → FramePacket for detect_batch
        packets_in = [
            FramePacket(
                frame_index=f.frame_index,
                timestamp_ms=f.timestamp_ms,
                image=f.image,
                width=f.width,
                height=f.height,
                triage_metrics={},
            )
            for f in batch
        ]

        packets_out = detector.detect_batch(
            packets_in, detector.confidence_threshold
        )

        for f in batch:
            accepted_frame_presence.append((f.frame_index, f.timestamp_ms, True))

        for pkt in packets_out:
            cd = pkt.corner_detection
            detection_rows.append(
                {
                    "detection_id": det_id,
                    "frame_index": pkt.frame_index,
                    "timestamp_ms": pkt.timestamp_ms,
                    "width": pkt.width,
                    "height": pkt.height,
                    "corners": [(float(p[0]), float(p[1])) for p in cd.corners],
                    "confidence": float(cd.confidence),
                    "source_frame_path": "",
                    "triage_metrics": {},
                }
            )
            det_id += 1

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
            "target_yolo_fps": None,
        },
        video_id=ctx.video_id,
    )
```

- [ ] **Step 2: Add CUDA branch to `_build_sampler_detector`**

In `_build_sampler_detector`, add an `elif` before the `else` block:

```python
    elif ctx.detector == "cuda":
        from card_capture.sampler.cuda_sampler import CudaSampler
        sampler = CudaSampler(
            video_path=_Path(ctx.video_path),
            stride=ctx.cuda_stride,
            opening_scan_s=ctx.opening_scan_s,
        )
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=ctx.corner_confidence,
            detection_width=640,
            device="cuda",   # explicit — hard-fail if CUDA unavailable on instance
        )
```

- [ ] **Step 3: Route through `_run_cuda_inference` in `detect.run()`**

In the `run()` function, after `sampler, detector = _build_sampler_detector(ctx)` and before the `_run_pipeline_workers` call, add the routing:

```python
    sampler, detector = _build_sampler_detector(ctx)
    options = _ctx_to_options(ctx, output_dir)

    if ctx.detector == "cuda":
        detect_out = _run_cuda_inference(ctx, sampler, detector, output_dir, frame_dir)
        _save_corner_samples(ctx, detect_out.detection_rows, output_dir)
        return detect_out

    stats, consumer_stats, raw_rows = _run_pipeline_workers(...)
```

- [ ] **Step 4: Run the 3 new tests**

```bash
python3 -m pytest tests/pipeline/test_detect_cuda.py -v 2>&1 | tail -10
```

Expected: all 3 PASS.

- [ ] **Step 5: Run broader suite**

```bash
python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py 2>&1 | tail -8
```

Expected: same pass/fail counts as before this task.

- [ ] **Step 6: Commit**

```bash
git add pipeline/steps/detect.py
git commit -m "feat(cuda): _run_cuda_inference + cuda branch in _build_sampler_detector"
```

---

### Task 6: Apply CUDA config in `vastai_worker.py`

**Files:**
- Modify: `app/vastai_worker.py`

- [ ] **Step 1: Read the current `_run_pipeline` function**

Read `app/vastai_worker.py` and find `_run_pipeline`. It currently calls `subprocess.run([sys.executable, "-m", "pipeline.card_capture_flow", ...])`.

- [ ] **Step 2: Add config override helpers and update `_run_pipeline`**

Add the following two functions before `_run_pipeline`:

```python
_CUDA_CONFIG_OVERRIDES: dict = {
    "detector": "cuda",
    "device": "cuda",
    "cuda_stride": 2,
    "cuda_batch_size": 32,
    "pipeline_backend": "cuda",
}

_CONFIG_PATH = Path(__file__).parent.parent / "card_capture_config.json"


def _apply_cuda_config() -> dict:
    """Write CUDA overrides to config; return original values for restore."""
    cfg: dict = {}
    if _CONFIG_PATH.exists():
        try:
            cfg = json.loads(_CONFIG_PATH.read_text())
        except Exception:
            pass
    original = {k: cfg.get(k) for k in _CUDA_CONFIG_OVERRIDES}
    cfg.update(_CUDA_CONFIG_OVERRIDES)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return original


def _restore_config(original: dict) -> None:
    """Restore config values that were overridden by _apply_cuda_config."""
    if not _CONFIG_PATH.exists():
        return
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        cfg.update(original)
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass
```

Then update `_run_pipeline` to wrap the subprocess call:

```python
def _run_pipeline(job: dict) -> None:
    job_id = job["job_id"]
    video_path = job["video_path"]
    config_preset = job.get("config_preset", "balanced")
    output_dir = _OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    original = _apply_cuda_config()
    try:
        repo_root = Path(__file__).parent.parent
        cmd = [
            sys.executable, "-m", "pipeline.card_capture_flow",
            "--no-pylint", "run",
            "--video", video_path,
            "--output-dir", str(output_dir),
            "--db", str(output_dir / "cards.sqlite"),
            "--config-preset", config_preset,
            "--ui-run-id", job_id,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-1000:] or result.stdout[-500:])
    finally:
        _restore_config(original)

    _package_results(job_id, output_dir, output_dir / "cards.sqlite")
```

- [ ] **Step 3: Run worker tests**

```bash
python3 -m pytest tests/app/test_vastai_worker.py -q 2>&1 | tail -5
```

Expected: all pass (config helpers don't affect the TestClient tests).

- [ ] **Step 4: Commit**

```bash
git add app/vastai_worker.py
git commit -m "feat(cuda): apply cuda config overrides in vastai_worker before pipeline"
```

---

### Task 7: Cloud GPU badge on runs page

**Files:**
- Modify: `app/web/src/routes/runs/[run_id]/+page.svelte`

- [ ] **Step 1: Read the current run header area**

Read `app/web/src/routes/runs/[run_id]/+page.svelte` and find the `<h1>` or run title area (around where `run.run_id` is displayed).

- [ ] **Step 2: Add the badge**

In the run detail header — near where the run_id is shown — add:

```svelte
{#if run.detect_telemetry?.yolo_device === 'cuda' || run.run_id?.startsWith('batch_')}
    <span class="cloud-badge">☁ Cloud GPU</span>
{/if}
```

In the `<style>` section, add:

```css
    .cloud-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #0078d4;
        border: 1px solid #b3d9f5;
        border-radius: 4px;
        padding: 0.1rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        vertical-align: middle;
        margin-left: 0.5rem;
    }
```

- [ ] **Step 3: Verify the app builds**

```bash
cd app/web && npm run build 2>&1 | tail -5
```

Expected: no errors.

- [ ] **Step 4: Commit and push**

```bash
git add app/web/src/routes/runs/\[run_id\]/+page.svelte
git commit -m "feat(cuda): cloud GPU badge on run detail page"
git push origin main
```

---

## Also: Vast.ai Template Setup Runbook

After the code is implemented, create the deployment doc:

```bash
# After completing all tasks above:
cat > docs/vastai-template-setup.md << 'EOF'
# Vast.ai Base Template Setup

## What to install on the base instance before saving the template

1. Start a vast.ai instance with a PyTorch+CUDA base image:
   `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`

2. Clone the repo:
   ```
   git clone https://github.com/jpglick/card-capture.git /workspace/card-capture
   ```

3. Install heavy dependencies (these are baked into the template):
   ```
   cd /workspace/card-capture
   pip install -e '.[model]'
   pip install decord ultralytics kornia
   ```

4. Download the YOLO model so it's cached:
   ```
   python -c "from card_capture.detectors import CardcaptorUltralyticsDetector; CardcaptorUltralyticsDetector(device='cuda')._load_model()"
   ```

5. Download DINOv2 so it's cached:
   ```
   python -c "from card_capture.ml.models.dino_embedder import DinoEmbedder; DinoEmbedder()"
   ```

6. Save the instance as a template in the vast.ai console.
   Copy the template ID into `card_capture_config.json` → `"vast_template_id"`.

## On each boot (handled automatically by vastai_worker startup script)

```bash
cd /workspace/card-capture && git pull origin main -q
pip install -e '.[app]' -q
uvicorn app.vastai_worker:app --host 0.0.0.0 --port 8765
```
EOF
git add docs/vastai-template-setup.md
git commit -m "docs: vast.ai template setup runbook"
git push origin main
```
