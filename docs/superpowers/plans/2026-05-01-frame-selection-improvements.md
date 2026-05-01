# Frame Selection Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cadence-based frame sampling with stability-based two-pass detection, add aspect-ratio and complexity quality scoring, downscale frames before YOLO inference, and stop early after the first good detection — producing a single high-quality still of the card front per video.

**Architecture:** Four changes applied in sequence — aspect-ratio and complexity components added to `QualityScorer`; a `StabilityBasedSampler` that runs a cheap 160px diff scan to find still windows and seeks to the sharpest frame in each; detection-width downscaling with polygon rescaling in `CardcaptorUltralyticsDetector`; early-stop logic in `VideoProcessor.process()` keyed on `detections_to_stop` and `quality_floor`. All changes are independently tested. Existing `VideoSampler` and cadence behaviour are preserved as a `--sampler raw` fallback.

**Tech Stack:** Python 3.9+, OpenCV (`cv2`), NumPy, pytest, `unittest.mock`, existing project layout at `src/card_capture/`.

---

## Chunk 1: Quality scoring and StabilityBasedSampler

### Task 1: Quality scorer additions

**Files:**
- Modify: `src/card_capture/scoring.py`
- Modify: `tests/test_scoring_selector.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_scoring_selector.py`

```python
def test_quality_scorer_penalizes_wrong_aspect_ratio():
    """Portrait (≈0.714 w/h) should score higher on aspect_ratio than a square (mid-flip)."""
    # Portrait: w=63, h=88 → ratio ≈ 0.716 ≈ CARD_RATIO
    portrait = np.zeros((88, 63, 3), dtype=np.uint8)
    portrait[5:83, 5:58] = 120
    # Square: w=88, h=88 → ratio 1.0 (foreshortened)
    square = np.zeros((88, 88, 3), dtype=np.uint8)
    square[5:83, 5:83] = 120

    scorer = QualityScorer(target_pixels=88 * 88)
    portrait_score = scorer.score(portrait, detection_confidence=0.9)
    square_score = scorer.score(square, detection_confidence=0.9)

    assert "aspect_ratio" in portrait_score.components
    assert portrait_score.components["aspect_ratio"] > square_score.components["aspect_ratio"]


def test_quality_scorer_complexity_rewards_textured_image():
    """Textured image (card artwork) should score higher on complexity than uniform (plain back)."""
    rng = np.random.default_rng(42)
    textured = rng.integers(0, 256, (100, 70, 3), dtype=np.uint8)
    uniform = np.full((100, 70, 3), 128, dtype=np.uint8)

    scorer = QualityScorer(target_pixels=100 * 70)
    textured_score = scorer.score(textured, detection_confidence=0.9)
    uniform_score = scorer.score(uniform, detection_confidence=0.9)

    assert "complexity" in textured_score.components
    assert textured_score.components["complexity"] > uniform_score.components["complexity"]


def test_quality_scorer_has_six_components():
    """Score components dict must contain all six keys after the rebalance."""
    image = np.full((88, 63, 3), 128, dtype=np.uint8)
    scorer = QualityScorer(target_pixels=88 * 63)
    score = scorer.score(image, detection_confidence=1.0)

    assert set(score.components.keys()) == {
        "sharpness", "glare", "aspect_ratio", "size", "complexity", "confidence"
    }
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/josh/WebstormProjects/vc2
PYTHONPATH=src python3 -m pytest tests/test_scoring_selector.py -v -k "aspect_ratio or complexity or six_components"
```

Expected: `FAILED` — `KeyError: 'aspect_ratio'` or similar.

- [ ] **Step 3: Replace `src/card_capture/scoring.py`**

```python
from __future__ import annotations

import cv2
import numpy as np

from .models import QualityScore

CARD_ASPECT_RATIO: float = 63.5 / 88.9  # ≈ 0.714 (width / height, standard trading card)
ASPECT_TOLERANCE: float = 0.25


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class QualityScorer:
    def __init__(self, target_pixels: int = 600 * 900):
        self.target_pixels = target_pixels

    def score(self, image: np.ndarray, detection_confidence: float) -> QualityScore:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = _clamp(laplacian_variance / 1000.0)

        overexposed_fraction = float((gray >= 245).mean())
        glare = _clamp(1.0 - overexposed_fraction * 4.0)

        h, w = image.shape[:2]
        actual_ratio = (w / h) if h > 0 else 0.0
        aspect_ratio = _clamp(
            1.0 - abs(actual_ratio - CARD_ASPECT_RATIO) / ASPECT_TOLERANCE
        )

        size = _clamp((h * w) / float(self.target_pixels))

        # Grayscale std-dev rewards textured fronts over plain backs
        complexity = _clamp(float(gray.std()) / 80.0)

        confidence = _clamp(float(detection_confidence))

        total = (
            sharpness * 0.30
            + glare * 0.20
            + aspect_ratio * 0.20
            + size * 0.15
            + complexity * 0.10
            + confidence * 0.05
        )
        components = {
            "sharpness": round(sharpness, 6),
            "glare": round(glare, 6),
            "aspect_ratio": round(aspect_ratio, 6),
            "size": round(size, 6),
            "complexity": round(complexity, 6),
            "confidence": round(confidence, 6),
        }
        return QualityScore(total=round(total, 6), components=components)
```

- [ ] **Step 4: Run all scorer tests**

```bash
PYTHONPATH=src python3 -m pytest tests/test_scoring_selector.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/scoring.py tests/test_scoring_selector.py
git commit -m "feat(scoring): add aspect_ratio and complexity quality components

Rebalance weights to sum to 1.0. New components:
- aspect_ratio (0.20): penalises mid-flip foreshortened crops using
  CARD_RATIO=0.714 and TOLERANCE=0.25
- complexity (0.10): grayscale std-dev rewards textured fronts over
  plain backs (normalised by 80.0)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: StabilityBasedSampler

**Files:**
- Modify: `src/card_capture/sampler.py`
- Create: `tests/test_sampler.py`

- [ ] **Step 1: Create `tests/test_sampler.py` with failing tests**

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from card_capture.models import FrameSample
from card_capture.sampler import StabilityBasedSampler, StableWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_video(tmp_path: Path, frames: list, fps: float = 30.0) -> Path:
    """Write a list of BGR numpy frames to a temporary .avi file."""
    path = tmp_path / "test.avi"
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), f"VideoWriter failed to open (codec unavailable?): {path}"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def gray_frames(count: int, value: int = 128) -> list:
    return [np.full((240, 320, 3), value, dtype=np.uint8) for _ in range(count)]


# ---------------------------------------------------------------------------
# _find_stable_windows
# ---------------------------------------------------------------------------

def test_finds_single_stable_window(tmp_path):
    """30 identical frames → one stable window detected."""
    path = make_video(tmp_path, gray_frames(30))
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    assert len(windows) == 1
    assert isinstance(windows[0], StableWindow)
    assert windows[0].best_frame_index >= 0


def test_best_frame_index_is_source_frame_number(tmp_path):
    """best_frame_index must refer to actual source video frame position
    (passable directly to cv2.CAP_PROP_POS_FRAMES), not the scan counter."""
    path = make_video(tmp_path, gray_frames(30), fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    assert len(windows) == 1
    # source_fps=30, scan_fps=10 → frame_step=3
    # frame 0 sets the initial diff reference; best_frame_index comes from
    # run_frames which starts accumulating at frame 3.
    valid_source_frames = set(range(3, 30, 3))
    assert windows[0].best_frame_index in valid_source_frames


def test_no_stable_windows_on_constant_motion(tmp_path):
    """Alternating black/white frames → no stable run reaches min_stable_frames."""
    frames = [
        np.full((240, 320, 3), 0 if i % 2 == 0 else 200, dtype=np.uint8)
        for i in range(30)
    ]
    path = make_video(tmp_path, frames)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    assert len(windows) == 0


def test_finds_two_stable_windows_separated_by_motion(tmp_path):
    """Two distinct stable periods separated by a high-motion transition → 2 windows."""
    frames = gray_frames(15, value=64) + gray_frames(15, value=200)
    path = make_video(tmp_path, frames, fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    assert len(windows) == 2
    assert windows[0].best_frame_index < windows[1].best_frame_index


def test_stable_window_dataclass_has_start_end_best(tmp_path):
    """StableWindow exposes start_frame, end_frame, best_frame_index."""
    path = make_video(tmp_path, gray_frames(30))
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    windows = sampler._find_stable_windows(path)

    w = windows[0]
    assert hasattr(w, "start_frame")
    assert hasattr(w, "end_frame")
    assert hasattr(w, "best_frame_index")
    assert w.start_frame <= w.best_frame_index <= w.end_frame


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------

def test_sample_yields_one_frame_sample_per_stable_window(tmp_path):
    """sample() yields exactly one FrameSample per stable window."""
    path = make_video(tmp_path, gray_frames(30), fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    results = list(sampler.sample(path, sample_fps=5.0))

    assert len(results) == 1
    s = results[0]
    assert isinstance(s, FrameSample)
    assert s.width == 320
    assert s.height == 240
    assert s.frame_index in set(range(0, 30, 3))


def test_sample_yields_empty_when_no_stable_windows(tmp_path):
    """sample() yields nothing if pass 1 finds no stable windows."""
    frames = [
        np.full((240, 320, 3), 0 if i % 2 == 0 else 200, dtype=np.uint8)
        for i in range(30)
    ]
    path = make_video(tmp_path, frames)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    results = list(sampler.sample(path, sample_fps=5.0))

    assert results == []


def test_sample_fps_argument_is_ignored(tmp_path):
    """sample_fps is accepted for interface compatibility but does not affect output."""
    path = make_video(tmp_path, gray_frames(30), fps=30.0)
    sampler = StabilityBasedSampler(
        scan_fps=10, scan_width=80, motion_threshold=8.0, min_stable_frames=3
    )
    results_5 = list(sampler.sample(path, sample_fps=5.0))
    results_30 = list(sampler.sample(path, sample_fps=30.0))

    assert len(results_5) == len(results_30) == 1
    assert results_5[0].frame_index == results_30[0].frame_index


def test_raises_on_missing_video(tmp_path):
    sampler = StabilityBasedSampler()
    with pytest.raises(ValueError, match="Could not decode video"):
        list(sampler.sample(tmp_path / "nonexistent.avi", sample_fps=5.0))
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=src python3 -m pytest tests/test_sampler.py -v
```

Expected: `ImportError: cannot import name 'StabilityBasedSampler'`

- [ ] **Step 3: Replace `src/card_capture/sampler.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .models import FrameSample


@dataclass
class StableWindow:
    """A contiguous run of low-motion frames found during pass 1.

    start_frame and end_frame are retained for logging/debugging.
    Only best_frame_index is used by pass 2 when seeking the video capture.
    All three values are source video frame numbers (0-based, usable with
    cv2.CAP_PROP_POS_FRAMES).
    """

    start_frame: int
    end_frame: int
    best_frame_index: int


class StabilityBasedSampler:
    """Two-pass sampler: cheap diff scan to find still windows, then seek to
    the sharpest frame in each window for full-resolution detection.

    Pass 1: decode at scan_fps with frames downscaled to scan_width wide.
            Compute per-frame pixel diff; track stable runs. Record the source
            frame number and Laplacian variance for each sampled frame — source
            frame numbers (not scan counters) are stored so they can be passed
            directly to cv2.CAP_PROP_POS_FRAMES.

    Pass 2: for each stable window, seek to best_frame_index and yield the
            full-resolution FrameSample.

    sample_fps is intentionally ignored — scan_fps is set via the constructor.
    The argument exists solely for interface compatibility with VideoSampler.
    """

    def __init__(
        self,
        scan_fps: float = 10.0,
        scan_width: int = 160,
        motion_threshold: float = 8.0,
        min_stable_frames: int = 5,
    ) -> None:
        self.scan_fps = scan_fps
        self.scan_width = scan_width
        self.motion_threshold = motion_threshold
        self.min_stable_frames = min_stable_frames

    def _find_stable_windows(self, video_path: Path) -> List[StableWindow]:
        """Pass 1: decode at low resolution and return stable window descriptors."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or self.scan_fps
        frame_step = max(1, int(round(source_fps / self.scan_fps)))

        windows: List[StableWindow] = []
        run_start: Optional[int] = None
        # Each entry: (source_frame_index, laplacian_variance)
        run_frames: List[tuple] = []
        prev_gray: Optional[np.ndarray] = None
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % frame_step == 0:
                    h, w = frame.shape[:2]
                    scaled_h = max(1, int(round(h * self.scan_width / w)))
                    small = cv2.resize(frame, (self.scan_width, scaled_h))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                    if prev_gray is not None:
                        diff = float(
                            np.abs(
                                gray.astype(np.float32) - prev_gray.astype(np.float32)
                            ).mean()
                        )
                        if diff < self.motion_threshold:
                            if run_start is None:
                                run_start = frame_index
                            run_frames.append((frame_index, lap_var))
                        else:
                            if (
                                run_start is not None
                                and len(run_frames) >= self.min_stable_frames
                            ):
                                best_idx = max(run_frames, key=lambda x: x[1])[0]
                                windows.append(
                                    StableWindow(
                                        start_frame=run_start,
                                        end_frame=run_frames[-1][0],
                                        best_frame_index=best_idx,
                                    )
                                )
                            run_start = None
                            run_frames = []

                    prev_gray = gray

                frame_index += 1

            # Flush any open run at end of video
            if run_start is not None and len(run_frames) >= self.min_stable_frames:
                best_idx = max(run_frames, key=lambda x: x[1])[0]
                windows.append(
                    StableWindow(
                        start_frame=run_start,
                        end_frame=run_frames[-1][0],
                        best_frame_index=best_idx,
                    )
                )
        finally:
            capture.release()

        return windows

    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:  # noqa: ARG002
        """Yield one FrameSample per stable window (the sharpest frame in each).

        sample_fps is intentionally unused — scan_fps is set via the constructor.
        """
        video_path = Path(video_path)
        windows = self._find_stable_windows(video_path)
        if not windows:
            return

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {video_path}")

        try:
            for window in windows:
                capture.set(cv2.CAP_PROP_POS_FRAMES, window.best_frame_index)
                # Read timestamp BEFORE capture.read() — OpenCV advances the
                # position counter on read, which would cause an off-by-one.
                timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                yield FrameSample(
                    frame_index=window.best_frame_index,
                    timestamp_ms=timestamp_ms,
                    image=frame,
                    width=width,
                    height=height,
                )
        finally:
            capture.release()


class VideoSampler:
    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or sample_fps
        frame_step = max(1, int(round(source_fps / sample_fps))) if sample_fps > 0 else 1
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_step == 0:
                    height, width = frame.shape[:2]
                    timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    yield FrameSample(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        image=frame,
                        width=width,
                        height=height,
                    )
                frame_index += 1
        finally:
            capture.release()


class SyntheticSampler:
    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:
        image = np.zeros((120, 90, 3), dtype=np.uint8)
        image[15:105, 10:80] = 180
        yield FrameSample(
            frame_index=0,
            timestamp_ms=0,
            image=image,
            width=90,
            height=120,
        )
```

- [ ] **Step 4: Run sampler tests**

```bash
PYTHONPATH=src python3 -m pytest tests/test_sampler.py -v
```

Expected: all pass.

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
PYTHONPATH=src python3 -m pytest -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/sampler.py tests/test_sampler.py
git commit -m "feat(sampler): add two-pass StabilityBasedSampler

Pass 1: cheap 160px diff scan identifies still windows; tracks actual
source frame numbers (not scan counters) and Laplacian sharpness.
Pass 2: seeks to the sharpest frame per window and yields full-res.

VideoSampler preserved unchanged as --sampler raw fallback.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Chunk 2: Detection downscaling, pipeline early stop, CLI, and running instructions

### Task 3: Detection downscaling

**Files:**
- Modify: `src/card_capture/detectors.py`
- Create: `tests/test_detectors.py`

- [ ] **Step 1: Create `tests/test_detectors.py` with failing tests**

```python
from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch

from card_capture.detectors import CardcaptorUltralyticsDetector
from card_capture.models import FrameSample


def _make_frame(height: int, width: int) -> FrameSample:
    return FrameSample(
        frame_index=0,
        timestamp_ms=0,
        image=np.zeros((height, width, 3), dtype=np.uint8),
        width=width,
        height=height,
    )


def _empty_model():
    """YOLO model mock that returns no detections."""
    m = MagicMock()
    m.return_value = []
    return m


def test_detector_downscales_wide_frame_before_inference():
    """Model receives a 640-wide image when original frame is wider."""
    detector = CardcaptorUltralyticsDetector(detection_width=640)
    frame = _make_frame(height=960, width=1280)

    model_mock = _empty_model()
    with patch.object(detector, "_load_model", return_value=model_mock):
        detector.detect(frame)

    passed_image = model_mock.call_args[0][0]
    assert passed_image.shape[1] == 640   # width
    assert passed_image.shape[0] == 480   # height: 960 * 640/1280 = 480


def test_detector_skips_resize_for_frame_already_small():
    """No resize when original_width <= detection_width."""
    detector = CardcaptorUltralyticsDetector(detection_width=640)
    frame = _make_frame(height=480, width=320)

    model_mock = _empty_model()
    with patch.object(detector, "_load_model", return_value=model_mock):
        detector.detect(frame)

    passed_image = model_mock.call_args[0][0]
    assert passed_image.shape[1] == 320
    assert passed_image.shape[0] == 480


def test_detector_rescales_polygon_to_original_frame_space():
    """Polygon coordinates from detection space must be scaled back to original
    frame dimensions using separate x and y scale factors."""
    detector = CardcaptorUltralyticsDetector(detection_width=640)
    # 1280x960 → detection at 640x480 → scale_x = scale_y = 2.0
    frame = _make_frame(height=960, width=1280)

    obb_mock = MagicMock()
    obb_mock.conf.cpu.return_value.numpy.return_value = np.array([0.9])
    obb_mock.cls.cpu.return_value.numpy.return_value = np.array([0])
    # Points in 640x480 detection space: a 100x100 square
    obb_mock.xyxyxyxy.cpu.return_value.numpy.return_value = np.array(
        [[[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]],
        dtype=np.float32,
    )
    result_mock = MagicMock()
    result_mock.obb = obb_mock
    model_mock = MagicMock()
    model_mock.return_value = [result_mock]

    with patch.object(detector, "_load_model", return_value=model_mock):
        detections = detector.detect(frame)

    assert len(detections) == 1
    poly = detections[0].polygon
    # scale_x = 1280/640 = 2.0, scale_y = 960/480 = 2.0
    assert poly[0] == (200.0, 200.0)
    assert poly[1] == (400.0, 200.0)
    assert poly[2] == (400.0, 400.0)
    assert poly[3] == (200.0, 400.0)
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=src python3 -m pytest tests/test_detectors.py -v
```

Expected: `FAILED` — polygon coords not rescaled, resize not applied.

- [ ] **Step 3: Update `CardcaptorUltralyticsDetector` in `src/card_capture/detectors.py`**

Replace the class (keep imports and `FakeCardDetector` and `CardDetector` protocol unchanged):

```python
class CardcaptorUltralyticsDetector:
    runtime = "ultralytics"
    model_name = "AlecKarfonta/cardcaptor-v3"

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        repo_id: str = "AlecKarfonta/cardcaptor-v3",
        filename: str = "weights/cardcaptor_v3_best.pt",
        detection_width: int = 640,
    ):
        self.confidence_threshold = confidence_threshold
        self.repo_id = repo_id
        self.filename = filename
        self.detection_width = detection_width
        self._model = None

    def detect(self, frame: FrameSample) -> List[CardDetection]:
        model = self._load_model()
        original_h, original_w = frame.image.shape[:2]

        if original_w > self.detection_width:
            scaled_w = self.detection_width
            scaled_h = max(1, int(round(original_h * self.detection_width / original_w)))
            detect_image = cv2.resize(frame.image, (scaled_w, scaled_h))
            scale_x = original_w / scaled_w
            scale_y = original_h / scaled_h
        else:
            detect_image = frame.image
            scale_x = 1.0
            scale_y = 1.0

        results = model(detect_image, conf=self.confidence_threshold, verbose=False)
        detections: List[CardDetection] = []
        for result in results:
            obb = getattr(result, "obb", None)
            if obb is None or obb.conf is None:
                continue
            polygons = obb.xyxyxyxy.cpu().numpy()
            confidences = obb.conf.cpu().numpy()
            labels = (
                obb.cls.cpu().numpy()
                if obb.cls is not None
                else [0] * len(confidences)
            )
            for polygon_array, confidence, label in zip(polygons, confidences, labels):
                confidence_float = float(confidence)
                if confidence_float < self.confidence_threshold:
                    continue
                polygon = tuple(
                    (float(point[0]) * scale_x, float(point[1]) * scale_y)
                    for point in polygon_array
                )
                if len(polygon) != 4:
                    continue
                detections.append(
                    CardDetection(
                        frame_index=frame.frame_index,
                        timestamp_ms=frame.timestamp_ms,
                        polygon=polygon,  # type: ignore[arg-type]
                        confidence=confidence_float,
                        metadata={
                            "runtime": self.runtime,
                            "model": self.model_name,
                            "class_id": int(label),
                        },
                    )
                )
        return detections

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from huggingface_hub import hf_hub_download
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Real card detection requires optional dependencies. "
                "Install with: pip install '.[model]'"
            ) from exc

        model_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
        self._model = YOLO(model_path)
        return self._model
```

Also add `import cv2` at the top of `detectors.py` (it is not currently imported there).

- [ ] **Step 4: Run detector tests**

```bash
PYTHONPATH=src python3 -m pytest tests/test_detectors.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/detectors.py tests/test_detectors.py
git commit -m "feat(detectors): downscale frames to detection_width before YOLO inference

Resize proportionally before inference; scale polygon coordinates back
to original frame space using separate x/y scale factors. Skip resize
when frame is already <= detection_width wide. Default detection_width=640.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Pipeline early stop

**Files:**
- Modify: `src/card_capture/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add failing tests to `tests/test_pipeline.py`**

The existing `FakeDetector` class (already in `test_pipeline.py`) is reused here. Add after the existing test:

```python
def test_early_stop_halts_after_first_qualifying_detection(tmp_path: Path):
    """Pipeline breaks out of the frame loop once detections_to_stop
    detections exceed quality_floor, without consuming further frames."""
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    processed_frames = []

    class CountingSampler:
        def sample(self, video_path, sample_fps):
            for i in range(5):
                processed_frames.append(i)
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                image[10:90, 10:90] = 180
                yield FrameSample(
                    frame_index=i,
                    timestamp_ms=i * 200,
                    image=image,
                    width=100,
                    height=100,
                )

    VideoProcessor(
        storage=storage,
        sampler=CountingSampler(),
        detector=FakeDetector(),
    ).process(
        video_path,
        ProcessingOptions(
            output_dir=tmp_path / "output",
            sample_fps=5,
            max_candidates=5,
            detections_to_stop=1,
            quality_floor=0.2,  # FakeDetector crop scores ≈ 0.249 > 0.2
        ),
    )

    # Generator is closed by `break` after the first frame — frame 1 never appended.
    assert len(processed_frames) == 1


def test_early_stop_disabled_when_zero(tmp_path: Path):
    """detections_to_stop=0 processes all sampled frames."""
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    processed_frames = []

    class CountingSampler:
        def sample(self, video_path, sample_fps):
            for i in range(3):
                processed_frames.append(i)
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                image[10:90, 10:90] = 180
                yield FrameSample(
                    frame_index=i,
                    timestamp_ms=i * 200,
                    image=image,
                    width=100,
                    height=100,
                )

    VideoProcessor(
        storage=storage,
        sampler=CountingSampler(),
        detector=FakeDetector(),
    ).process(
        video_path,
        ProcessingOptions(
            output_dir=tmp_path / "output",
            sample_fps=5,
            max_candidates=5,
            detections_to_stop=0,
            quality_floor=0.2,
        ),
    )

    assert len(processed_frames) == 3
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=src python3 -m pytest tests/test_pipeline.py -v -k "early_stop"
```

Expected: `FAILED` — `TypeError: ProcessingOptions() got unexpected keyword argument 'detections_to_stop'`

- [ ] **Step 3: Update `src/card_capture/pipeline.py`**

**Change 1** — update `ProcessingOptions` dataclass:

```python
@dataclass(frozen=True)
class ProcessingOptions:
    output_dir: Path
    sample_fps: float = 5.0
    max_candidates: int = 10
    confidence_threshold: float = 0.25
    group_gap_ms: int = 1000
    detections_to_stop: int = 1
    quality_floor: float = 0.5
```

**Change 2** — replace the frame-processing loop inside `VideoProcessor.process()` (the block starting `candidates: List[ScoredCandidate] = []` through `detection_count += 1`):

```python
        candidates: List[ScoredCandidate] = []
        detection_count = 0
        good_detection_count = 0

        for frame in self.sampler.sample(video_path, options.sample_fps):
            source_frame_path = frame_dir / f"video_{video_id}_frame_{frame.frame_index}.jpg"
            cv2.imwrite(str(source_frame_path), frame.image)

            stop_this_frame = False
            for detection in self.detector.detect(frame):
                if detection.confidence < options.confidence_threshold:
                    continue
                crop = self.cropper.crop(frame.image, detection.polygon)
                score = self.scorer.score(crop.image, detection.confidence)
                crop_path = crop_dir / (
                    f"video_{video_id}_frame_{frame.frame_index}_det_{detection_count}.jpg"
                )
                cv2.imwrite(str(crop_path), crop.image)
                detection_id = self.storage.add_detection(
                    video_id=video_id,
                    detection=detection,
                    crop_path=str(crop_path),
                    source_frame_path=str(source_frame_path),
                    score=score,
                    crop_width=crop.width,
                    crop_height=crop.height,
                )
                candidates.append(
                    ScoredCandidate(
                        detection_id=detection_id,
                        timestamp_ms=detection.timestamp_ms,
                        image_path=str(crop_path),
                        score=score,
                    )
                )
                detection_count += 1
                if (
                    options.detections_to_stop > 0
                    and score.total >= options.quality_floor
                ):
                    good_detection_count += 1
                    if good_detection_count >= options.detections_to_stop:
                        stop_this_frame = True
                        break  # stop processing further detections in this frame

            if stop_this_frame:
                break  # stop consuming more frames; Python auto-closes the generator
```

- [ ] **Step 4: Run all pipeline tests**

```bash
PYTHONPATH=src python3 -m pytest tests/test_pipeline.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): add early stop after first quality detection

New ProcessingOptions fields:
  detections_to_stop (default 1): halt after this many score >= quality_floor
  quality_floor (default 0.5): minimum score to count toward early stop

Set detections_to_stop=0 to disable. Breaking the frame loop auto-closes
the sampler generator via Python's GeneratorExit mechanism.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: CLI flags

**Files:**
- Modify: `src/card_capture/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing test to `tests/test_cli.py`**

```python
def test_process_subparser_accepts_new_flags():
    from card_capture.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "process", "video.mov",
        "--detection-width", "320",
        "--scan-fps", "5",
        "--scan-width", "120",
        "--motion-threshold", "12.0",
        "--min-stable-frames", "4",
        "--sampler", "stability",
        "--detections-to-stop", "2",
        "--quality-floor", "0.6",
    ])
    assert args.detection_width == 320
    assert args.scan_fps == 5.0
    assert args.scan_width == 120
    assert args.motion_threshold == 12.0
    assert args.min_stable_frames == 4
    assert args.sampler == "stability"
    assert args.detections_to_stop == 2
    assert args.quality_floor == 0.6
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=src python3 -m pytest tests/test_cli.py::test_process_subparser_accepts_new_flags -v
```

Expected: `FAILED` — `error: unrecognized arguments: --detection-width`

- [ ] **Step 3: Replace `src/card_capture/cli.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .detectors import CardcaptorUltralyticsDetector, FakeCardDetector
from .pipeline import ProcessingOptions, VideoProcessor
from .sampler import StabilityBasedSampler, SyntheticSampler, VideoSampler
from .storage import Storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="card-capture")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process a local video file")
    process.add_argument("video_path", type=Path)
    process.add_argument("--output-dir", type=Path, default=Path("card_capture_output"))
    process.add_argument("--db", type=Path, default=Path("card_capture_output/cards.sqlite"))
    process.add_argument("--sample-fps", type=float, default=5.0)
    process.add_argument("--max-candidates", type=int, default=10)
    process.add_argument("--confidence", type=float, default=0.25)
    process.add_argument(
        "--detector",
        choices=["cardcaptor", "fake"],
        default="cardcaptor",
        help="cardcaptor for real detection, fake for smoke tests",
    )
    process.add_argument(
        "--sampler",
        choices=["stability", "raw"],
        default="stability",
        help="stability (default): two-pass stability sampler; raw: cadence-based VideoSampler",
    )
    process.add_argument(
        "--scan-fps", type=float, default=10.0,
        help="Pass-1 scan cadence in frames per second (default: 10)",
    )
    process.add_argument(
        "--scan-width", type=int, default=160,
        help="Pass-1 scan frame width in pixels (default: 160)",
    )
    process.add_argument(
        "--motion-threshold", type=float, default=8.0,
        help="Max mean pixel diff (0-255) to count as stable (default: 8.0)",
    )
    process.add_argument(
        "--min-stable-frames", type=int, default=5,
        help="Min consecutive stable scan frames to form a window (default: 5)",
    )
    process.add_argument(
        "--detection-width", type=int, default=640,
        help="Frame width passed to YOLO detector, proportionally scaled (default: 640)",
    )
    process.add_argument(
        "--detections-to-stop", type=int, default=1,
        help="Stop after this many quality detections; 0 = disabled (default: 1)",
    )
    process.add_argument(
        "--quality-floor", type=float, default=0.5,
        help="Minimum quality score to count toward early stop (default: 0.5)",
    )

    review = subparsers.add_parser("review", help="Start the local review UI")
    review.add_argument("--db", type=Path, default=Path("card_capture_output/cards.sqlite"))
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "process":
        return _run_process(args)
    if args.command == "review":
        return _run_review(args)
    parser.error("unknown command")
    return 2


def _run_process(args: argparse.Namespace) -> int:
    storage = Storage(args.db)
    storage.initialize()

    if args.detector == "fake":
        detector = FakeCardDetector()
        sampler = SyntheticSampler()
    else:
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=args.confidence,
            detection_width=args.detection_width,
        )
        if args.sampler == "raw":
            sampler = VideoSampler()
        else:
            sampler = StabilityBasedSampler(
                scan_fps=args.scan_fps,
                scan_width=args.scan_width,
                motion_threshold=args.motion_threshold,
                min_stable_frames=args.min_stable_frames,
            )

    processor = VideoProcessor(storage=storage, sampler=sampler, detector=detector)
    result = processor.process(
        args.video_path,
        ProcessingOptions(
            output_dir=args.output_dir,
            sample_fps=args.sample_fps,
            max_candidates=args.max_candidates,
            confidence_threshold=args.confidence,
            detections_to_stop=args.detections_to_stop,
            quality_floor=args.quality_floor,
        ),
    )
    print(
        f"Processed video_id={result.video_id}: "
        f"{result.detection_count} detections, {result.saved_count} saved"
    )
    return 0


def _run_review(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Review UI requires: pip install '.[review]'") from exc

    from .review import create_app

    app = create_app(args.db)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run all CLI tests**

```bash
PYTHONPATH=src python3 -m pytest tests/test_cli.py -v
```

Expected: all pass.

- [ ] **Step 5: Run the complete test suite**

```bash
PYTHONPATH=src python3 -m pytest -v
```

Expected: all tests pass. Count should exceed the original 12.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/cli.py tests/test_cli.py
git commit -m "feat(cli): add flags for stability sampler, detection width, and early stop

New flags on the process command:
  --sampler (stability|raw)
  --scan-fps, --scan-width, --motion-threshold, --min-stable-frames
  --detection-width
  --detections-to-stop, --quality-floor

--detector fake always uses SyntheticSampler regardless of --sampler.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Verify and document running instructions

- [ ] **Step 1: Install model dependencies (first-time only)**

```bash
cd /Users/josh/WebstormProjects/vc2
python3 -m pip install '.[model]'
```

This installs `ultralytics` and `huggingface_hub`. The model weights (~25 MB) are downloaded on first run.

- [ ] **Step 2: Run the full test suite one final time**

```bash
PYTHONPATH=src python3 -m pytest -v
```

Expected: all tests pass. Record the final count.

- [ ] **Step 3: Basic run (recommended defaults)**

```bash
PYTHONPATH=src python3 -m card_capture.cli process /path/to/card_video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite
```

Defaults: stability sampler, stop after first quality detection (`quality_floor=0.5`).

- [ ] **Step 4: Tuned run for 10–60s single-card clips**

```bash
PYTHONPATH=src python3 -m card_capture.cli process /path/to/card_video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite \
  --sampler stability \
  --scan-fps 10 \
  --scan-width 160 \
  --motion-threshold 8.0 \
  --min-stable-frames 5 \
  --detection-width 640 \
  --detections-to-stop 1 \
  --quality-floor 0.5 \
  --confidence 0.25 \
  --max-candidates 10
```

- [ ] **Step 5: Fastest mode (smaller detection frame, trades accuracy for speed)**

```bash
PYTHONPATH=src python3 -m card_capture.cli process /path/to/card_video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite \
  --detection-width 320
```

- [ ] **Step 6: Thorough mode (lower motion threshold, no early stop)**

Useful when the card is briefly moving throughout or for videos with multiple stable phases:

```bash
PYTHONPATH=src python3 -m card_capture.cli process /path/to/card_video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite \
  --motion-threshold 5.0 \
  --min-stable-frames 8 \
  --detections-to-stop 0
```

- [ ] **Step 7: Fallback to old cadence-based sampling**

```bash
PYTHONPATH=src python3 -m card_capture.cli process /path/to/card_video.mov \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite \
  --sampler raw \
  --sample-fps 5
```

- [ ] **Step 8: Launch the review UI**

```bash
PYTHONPATH=src python3 -m card_capture.cli review \
  --db card_capture_output/cards.sqlite
```

Open **http://127.0.0.1:8000** to accept/reject captures.

Output layout:
```
card_capture_output/
  crops/    — all detected card crops (per-frame)
  best/     — top candidates chosen by the selector
  frames/   — source frames at each detection point
  cards.sqlite
```
