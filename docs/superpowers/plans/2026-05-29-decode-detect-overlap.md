# Decode↔Detect Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap CPU video decode with CoreML detection so the GPU/ANE works during the ~36 s `sample` stage instead of after it, cutting total pipeline time by ~min(decode, detect).

**Architecture:** A `FrameProducer` runs `StrideSampler.sample()` on a daemon thread, pushing `FrameSample`s onto a bounded queue. The `sample` stage starts the producer and returns after the first frame; the `detect` stage drains the queue, batches, runs inference, and populates `state["sampled_frames"]`. Stages stay distinct (no `_STAGES`/telemetry/sparkline changes). Separately, the dead `hwaccel` line is removed.

**Tech Stack:** Python 3.9, PyAV (software HEVC decode), `threading`/`queue`, pytest. Design spec: `docs/superpowers/specs/2026-05-29-decode-detect-overlap-design.md`.

---

## File Structure

- **Modify** `src/card_capture/sampler/__init__.py` — remove the no-op VideoToolbox branch in `_open_pyav_container` (`:223-233`).
- **Create** `src/card_capture/sampler/frame_producer.py` — `FrameProducer` (decode thread + bounded queue).
- **Modify** `src/card_capture/pipeline/stages/sample.py` — start the producer instead of `list(sampler.sample())`.
- **Modify** `src/card_capture/pipeline/stages/detect.py` — drain the producer instead of reading `state["sampled_frames"]`.
- **Create** `tests/sampler/test_pyav_open.py` — Task 1 test.
- **Create** `tests/sampler/test_frame_producer.py` — Task 2 tests.
- **Create** `tests/pipeline/stages/test_sample_stage.py` — Task 3 test.
- **Create** `tests/pipeline/stages/test_sample_detect_overlap.py` — Task 4 integration test.

---

## Task 1: Drop the no-op `hwaccel` videotoolbox line

**Files:**
- Modify: `src/card_capture/sampler/__init__.py:223-233`
- Test: `tests/sampler/test_pyav_open.py`

- [ ] **Step 1: Write the failing test**

Create `tests/sampler/test_pyav_open.py`:

```python
"""_open_pyav_container must use plain software decode (no hwaccel option)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from card_capture.sampler import VideoSampler


def test_open_pyav_container_uses_software_no_hwaccel():
    with patch("av.open") as mock_open:
        mock_open.return_value = MagicMock(name="container")
        container, hw = VideoSampler._open_pyav_container(Path("/tmp/x.mov"))

    assert hw is False
    assert mock_open.call_count == 1
    args, kwargs = mock_open.call_args
    assert "options" not in kwargs        # no {"hwaccel": ...}
    assert args == ("/tmp/x.mov",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/sampler/test_pyav_open.py -v`
Expected (on macOS): FAIL — current code passes `options={"hwaccel": "videotoolbox"}` and returns `hw=True`.

- [ ] **Step 3: Make the change**

In `src/card_capture/sampler/__init__.py`, replace the whole `_open_pyav_container` method (currently lines 223-233):

```python
    @staticmethod
    def _open_pyav_container(video_path: Path):
        """Open a PyAV container, preferring VideoToolbox hw-decode on macOS."""
        import av
        if platform.system() == "Darwin":
            try:
                container = av.open(str(video_path), options={"hwaccel": "videotoolbox"})
                return container, True
            except Exception:
                pass
        return av.open(str(video_path)), False
```

with:

```python
    @staticmethod
    def _open_pyav_container(video_path: Path):
        """Open a PyAV container with the software decoder.

        We previously passed ``options={"hwaccel": "videotoolbox"}`` here, but
        ``av.open`` options are demuxer (AVFormatContext) options — ``hwaccel`` is
        an ffmpeg-CLI concept, not a libav option key, so it was silently ignored
        and we ran software decode anyway while logging a false "videotoolbox".
        On Apple Silicon the multithreaded software HEVC decoder (~242 fps @ 4K10)
        beats single-stream VideoToolbox (~199 fps), so software is correct here.
        Returns ``(container, hw_active=False)``.
        """
        import av
        return av.open(str(video_path)), False
```

Then change the log line at `:238` from:

```python
        print(f"[sampler] pyav decoder={'videotoolbox' if hw else 'software'} format={pixel_format}", flush=True)
```

to:

```python
        print(f"[sampler] pyav decoder=software format={pixel_format}", flush=True)
```

(`platform` is still used at `:122`, so leave `import platform`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/sampler/test_pyav_open.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/sampler/__init__.py tests/sampler/test_pyav_open.py
git commit -m "fix(sampler): drop no-op videotoolbox hwaccel request (was silently software)"
```

---

## Task 2: `FrameProducer` (decode thread + bounded queue)

**Files:**
- Create: `src/card_capture/sampler/frame_producer.py`
- Test: `tests/sampler/test_frame_producer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/sampler/test_frame_producer.py`:

```python
"""FrameProducer: order, first-frame gating, error propagation, early stop."""
import time

import numpy as np
import pytest

from card_capture.models import FrameSample
from card_capture.sampler.frame_producer import FrameProducer


def _frame(i: int) -> FrameSample:
    return FrameSample(
        frame_index=i, timestamp_ms=i * 33,
        image=np.zeros((2, 2, 3), np.uint8), width=2, height=2,
    )


class _FakeSampler:
    def __init__(self, n, *, boom_at=None, delay=0.0):
        self.n = n
        self.boom_at = boom_at
        self.delay = delay

    def sample(self):
        for i in range(self.n):
            if self.boom_at is not None and i == self.boom_at:
                raise RuntimeError("decode failed")
            if self.delay:
                time.sleep(self.delay)
            yield _frame(i)


def test_yields_all_frames_in_order():
    prod = FrameProducer(_FakeSampler(5)).start()
    assert [f.frame_index for f in prod] == [0, 1, 2, 3, 4]


def test_wait_first_unblocks_then_drains_all():
    prod = FrameProducer(_FakeSampler(3, delay=0.01)).start()
    assert prod.wait_first(timeout=2.0) is True
    assert [f.frame_index for f in prod] == [0, 1, 2]


def test_producer_error_propagates_to_consumer():
    prod = FrameProducer(_FakeSampler(5, boom_at=2)).start()
    with pytest.raises(RuntimeError, match="decode failed"):
        list(prod)


def test_stop_terminates_blocked_producer_without_hanging():
    # maxsize=1 makes the producer block on put almost immediately.
    prod = FrameProducer(_FakeSampler(10_000), maxsize=1).start()
    assert prod.wait_first(timeout=2.0) is True
    prod.stop(timeout=2.0)
    assert prod._thread is not None and not prod._thread.is_alive()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/sampler/test_frame_producer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'card_capture.sampler.frame_producer'`

- [ ] **Step 3: Write the implementation**

Create `src/card_capture/sampler/frame_producer.py`:

```python
"""Background frame producer: decode on a worker thread, consume on the main thread.

The ``sample`` stage starts a :class:`FrameProducer`, which runs a sampler's
``.sample()`` (CPU/ffmpeg decode) on a daemon thread and pushes each
``FrameSample`` onto a bounded queue. The ``detect`` stage then drains the queue
and runs inference, so decode overlaps inference instead of completing first.
See docs/superpowers/specs/2026-05-29-decode-detect-overlap-design.md.
"""
from __future__ import annotations

import queue
import threading
from typing import Iterator, Optional, Protocol

from card_capture.models import FrameSample


class _Samplerish(Protocol):
    def sample(self) -> Iterator[FrameSample]: ...


_SENTINEL = object()


class FrameProducer:
    """Run ``sampler.sample()`` on a daemon thread, feeding a bounded queue."""

    def __init__(self, sampler: _Samplerish, *, maxsize: int = 32) -> None:
        self._sampler = sampler
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._first_ready = threading.Event()

    @property
    def queue(self) -> "queue.Queue":
        return self._queue

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def start(self) -> "FrameProducer":
        self._thread = threading.Thread(
            target=self._run, name="frame-producer", daemon=True
        )
        self._thread.start()
        return self

    def wait_first(self, timeout: Optional[float] = None) -> bool:
        """Block until the first frame is enqueued (or the run ended)."""
        return self._first_ready.wait(timeout)

    def _run(self) -> None:
        try:
            for frame in self._sampler.sample():
                if self._stop.is_set():
                    break
                self._queue.put(frame)
                self._first_ready.set()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer
            self._error = exc
        finally:
            self._first_ready.set()
            self._queue.put(_SENTINEL)

    def __iter__(self) -> Iterator[FrameSample]:
        """Yield frames until the sentinel, then join and re-raise producer errors."""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            yield item
        self.join()
        if self._error is not None:
            raise self._error

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the producer to stop, draining so a blocked ``put`` unblocks."""
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        for _ in range(max(1, int(timeout / 0.05))):
            if not thread.is_alive():
                break
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            thread.join(timeout=0.05)

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/sampler/test_frame_producer.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/sampler/frame_producer.py tests/sampler/test_frame_producer.py
git commit -m "feat(sampler): add FrameProducer for threaded decode handoff"
```

---

## Task 3: `sample` stage starts the producer

**Files:**
- Modify: `src/card_capture/pipeline/stages/sample.py` (replace `run`)
- Test: `tests/pipeline/stages/test_sample_stage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/stages/test_sample_stage.py`:

```python
"""sample stage starts a FrameProducer and does not drain it."""
from unittest.mock import MagicMock

from card_capture.pipeline.stages import sample as sample_stage
from card_capture.sampler.frame_producer import FrameProducer


def test_sample_starts_producer_without_draining(synthetic_two_cards_mov):
    request = MagicMock()
    request.input_video = str(synthetic_two_cards_mov)
    state = {"request": request}

    sample_stage.run(state, telemetry=MagicMock())

    assert isinstance(state["frame_producer"], FrameProducer)
    assert state["sampled_frames"] == []          # not drained in this stage
    assert state["estimated_frame_total"] > 0
    assert state["video_path"] == str(synthetic_two_cards_mov)

    frames = list(state["frame_producer"])         # draining works
    assert len(frames) > 0
    assert [f.frame_index for f in frames] == sorted(f.frame_index for f in frames)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/pipeline/stages/test_sample_stage.py -v`
Expected: FAIL with `KeyError: 'frame_producer'` (current `run` sets `sampled_frames` to a materialized list, not a producer).

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `src/card_capture/pipeline/stages/sample.py` with:

```python
"""Stage 1: Adaptive Presence Sampler (streaming producer).

Starts a background decode thread (``FrameProducer``) so the ``detect`` stage can
run inference while later frames are still decoding. Frames are NOT materialized
here; ``detect`` drains the producer and fills ``state["sampled_frames"]``.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from card_capture.sampler import StrideSampler
from card_capture.sampler.frame_producer import FrameProducer


def _estimate_selected_count(video_path: str, target_fps: float) -> int:
    """Cheap header probe -> estimated kept-frame count (for detect progress)."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return 0
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if src_fps <= 0 or total <= 0 or target_fps <= 0:
        return 0
    stride = max(1, round(src_fps / target_fps))
    return int(total // stride)


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    video_path = request.input_video.replace("artifact://local/", "")
    telemetry.resource_sample({"event": "decode_open", "path": video_path})

    sampler = StrideSampler(video_path=Path(video_path))
    producer = FrameProducer(sampler).start()
    # Block until the first frame is decoded so this stage's timing reflects
    # decode startup; all remaining frames overlap the detect stage.
    producer.wait_first(timeout=60.0)

    state["sampler"] = sampler
    state["video_path"] = video_path
    state["frame_producer"] = producer
    state["sampled_frames"] = []  # filled by the detect stage as it drains
    state["estimated_frame_total"] = _estimate_selected_count(
        video_path, sampler.target_yolo_fps
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/pipeline/stages/test_sample_stage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/sample.py tests/pipeline/stages/test_sample_stage.py
git commit -m "feat(sample): start FrameProducer instead of materializing all frames"
```

---

## Task 4: `detect` stage drains the producer

**Files:**
- Modify: `src/card_capture/pipeline/stages/detect.py` (replace `run`)
- Test: `tests/pipeline/stages/test_sample_detect_overlap.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/pipeline/stages/test_sample_detect_overlap.py`:

```python
"""sample+detect overlap: frame parity vs a direct sampler pass; one det/frame."""
from unittest.mock import MagicMock

from card_capture.pipeline.stages import detect as detect_stage
from card_capture.pipeline.stages import sample as sample_stage
from card_capture.sampler import StrideSampler


def test_sample_detect_overlap_parity(synthetic_two_cards_mov):
    request = MagicMock()
    request.input_video = str(synthetic_two_cards_mov)
    request.config = {"detector": "fake"}
    state = {"request": request}

    sample_stage.run(state, telemetry=MagicMock())
    detect_stage.run(state, telemetry=MagicMock())

    sampled = state["sampled_frames"]
    detections = state["detections"]

    expected = list(StrideSampler(video_path=synthetic_two_cards_mov).sample())
    assert len(sampled) > 0
    assert [f.frame_index for f in sampled] == [f.frame_index for f in expected]
    assert [f.timestamp_ms for f in sampled] == [f.timestamp_ms for f in expected]

    # FakeCardDetector emits exactly one DetectionPacket per frame.
    assert len(detections) == len(sampled)
    assert [d["frame_index"] for d in detections] == [f.frame_index for f in sampled]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/pipeline/stages/test_sample_detect_overlap.py -v`
Expected: FAIL — current `detect.run` reads `state["sampled_frames"]` (now `[]`), so `detections` is empty and the length assertion fails.

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `src/card_capture/pipeline/stages/detect.py` with:

```python
"""Stage 3: YOLO Corner Detection (drains the FrameProducer).

Consumes frames from ``state["frame_producer"]`` as they decode (overlapping
decode with inference), appends each to ``state["sampled_frames"]`` for
downstream stages, batches them, and runs the detector. Loads the model once.
"""
from __future__ import annotations

from card_capture.detectors import FakeCardDetector, CardcaptorUltralyticsDetector, probe_torch_device_status
from card_capture.models import FramePacket


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    config = request.config

    if "yolo_model" not in state:
        telemetry.resource_sample({"event": "model_load", "model": "yolo_obb"})
        detector_name = config.get("detector", "fake")
        if detector_name == "fake":
            detector = FakeCardDetector()
        else:
            device = config.get("device", "auto")
            device_status = probe_torch_device_status(device)
            detector = CardcaptorUltralyticsDetector(
                confidence_threshold=config.get("corner_confidence", 0.5),
                detection_width=config.get("detection_width", 640),
                device=device_status.resolved,
            )
        state["yolo_model"] = detector

    detector = state["yolo_model"]
    conf = config.get("corner_confidence", 0.5)
    producer = state["frame_producer"]
    frames_out = state["sampled_frames"]
    estimated_total = max(1, int(state.get("estimated_frame_total", 0)) or 1)

    batch_size = 16
    detections = []
    batch: list = []
    processed = 0

    def _flush() -> None:
        nonlocal processed
        if not batch:
            return
        packets = [
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
        detections.extend(detector.detect_batch(packets, conf))
        processed += len(batch)
        pct = min(99, int(100 * processed / estimated_total))
        telemetry.progress("detect", pct, f"{processed} frames")
        batch.clear()

    try:
        for frame in producer:        # overlaps decode with inference
            frames_out.append(frame)
            batch.append(frame)
            if len(batch) >= batch_size:
                _flush()
        _flush()
    finally:
        producer.stop()               # guarantee the decode thread ends

    telemetry.progress("detect", 100, f"{processed} frames")

    rows = []
    for i, p in enumerate(detections):
        rows.append({
            "detection_id": i + 1,
            "frame_index": p.frame_index,
            "timestamp_ms": p.timestamp_ms,
            "width": p.width,
            "height": p.height,
            "corners": p.corner_detection.corners,
            "confidence": p.corner_detection.confidence,
            "triage_metrics": {},
        })

    state["detections"] = rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/pipeline/stages/test_sample_detect_overlap.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline/stages/detect.py tests/pipeline/stages/test_sample_detect_overlap.py
git commit -m "feat(detect): drain FrameProducer to overlap decode with inference"
```

---

## Task 5: Full regression

**Files:** none (verification only)

- [ ] **Step 1: Run the pipeline + sampler suites**

Run: `python3 -m pytest tests/pipeline tests/sampler tests/test_sampler.py -q`
Expected: PASS (these directly exercise the changed stages/sampler).

- [ ] **Step 2: Run the full non-quarantine suite**

Run: `python3 -m pytest tests/ -m "not quarantine" -q`
Expected: PASS. If any existing test asserted the old `sample`/`detect` contract (e.g. `sampled_frames` populated by `sample`, or `decoder=videotoolbox`), update it to the streaming contract: `sample` now sets `frame_producer` + empty `sampled_frames`; `detect` fills `sampled_frames`. Re-run until green.

- [ ] **Step 3: Manual smoke (optional, real video)**

Run the web app (`bash` dev script or `uvicorn app.main:app ...`), process a `.MOV`, and confirm in the run log: the `[sampler] pyav decoder=software ...` line (no `videotoolbox`), and that `detect` progress advances while decoding (GPU/ANE busy during the formerly-idle window).

- [ ] **Step 4: Commit any test fixups**

```bash
git add -A
git commit -m "test: update sample/detect tests for streaming-overlap contract"
```

---

## Notes / Out of Scope

- Memory peak unchanged: full-res frames still accumulate in `state["sampled_frames"]` because `refine` warps from them. The bounded queue only caps the producer's lead. A crop-cache to bound the peak is a separate, deferred effort (see spec Non-Goals).
- No `_STAGES`, telemetry-layer, or `PipelineSparkline.svelte` changes: `sample` and `detect` remain distinct stages; `sample`'s bar shrinks to decode-startup and `detect`'s grows to absorb the overlapped window.
- `state["sampler"]` is preserved for parity/debugging but is not read by any downstream stage (verified).
