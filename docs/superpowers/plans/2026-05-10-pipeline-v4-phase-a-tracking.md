# Pipeline V4 — Phase A: Detection & Tracking Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `HysteresisTracker` with ByteTrack, adopt per-video adaptive session gap, delete the area-drop hard rule, and replace Stage-1 pixel-stat presence detection with a tiny MobileNetV3-Small visual classifier. Each change is verified against the regression harness from Spec 0.

**Architecture:** ByteTrack adapter wraps the `supervision` library to produce the same `TrackState`-like output the rest of the pipeline expects. Adaptive session gap is computed during Pass 1 of the existing `AdaptivePresenceSampler` from the inter-window gap distribution. The presence classifier is a small MobileNetV3-Small, trained on bootstrap data mined from existing pipeline detections; inference happens on the same scan-resolution proxy frames the sampler already produces.

**Tech Stack:** Python 3.9+, PyTorch 2.x with MPS, torchvision (MobileNetV3-Small), `supervision` library (ByteTrack), pytest. Adds `supervision` and `torchvision` to runtime deps.

**Prerequisite:** Spec 0 plan must be complete. Harness baseline `reports/baseline_v3.json` must exist.

---

## File Structure

**Create:**
- `src/card_capture/tracking/__init__.py` — exposes `ByteTrackAdapter`
- `src/card_capture/tracking/bytetrack_adapter.py` — wraps `supervision.ByteTrack`
- `src/card_capture/presence/__init__.py`
- `src/card_capture/presence/classifier.py` — MobileNetV3-Small wrapper, load + inference
- `src/card_capture/presence/training_data.py` — mines positives/negatives from existing pipeline runs
- `src/card_capture/train/__init__.py`
- `src/card_capture/train/presence.py` — training script (`python -m card_capture.train.presence ...`)
- `src/card_capture/adaptive_gap.py` — compute per-video session gap from Pass 1 stats
- `models/.gitkeep`
- `tests/test_bytetrack_adapter.py`
- `tests/test_adaptive_gap.py`
- `tests/test_presence_classifier.py`
- `tests/test_presence_training_data.py`

**Modify:**
- `pyproject.toml` — add `supervision`, `torchvision` runtime deps
- `src/card_capture/pipeline.py` — swap `HysteresisTracker` for `ByteTrackAdapter`; consume adaptive gap; remove fixed `null_patience_frames` use as session boundary (keep as max-bound)
- `src/card_capture/sampler.py` — replace Otsu thresholding in `AdaptivePresenceSampler._build_windows` with classifier output; expose inter-window gap distribution
- `src/card_capture/selector.py` — delete `HysteresisTracker` class and its `detect_flip` method (keep `CandidateSelector`)
- `tests/test_hysteresis_tracker.py` — delete (covered tracker is gone)

**Delete:**
- `src/card_capture/selector.py:32-233` — `HysteresisTracker` class + helpers exclusive to it (keep `_calculate_centroid`, `_euclidean_distance`, `_get_polygon_area` if used elsewhere; verify with `grep`)
- `tests/test_hysteresis_tracker.py` — entire file

---

## Task 1: Add ByteTrack dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit pyproject.toml**

Append `supervision` and `torchvision` to the `dependencies` array under `[project]`:

```toml
dependencies = [
  "numpy",
  "opencv-python",
  "supervision>=0.21",
  "torchvision",
]
```

- [ ] **Step 2: Install**

```bash
pip install -e .
```

Expected: pip resolves and installs `supervision` and `torchvision`. No errors.

- [ ] **Step 3: Smoke test the import**

```bash
python -c "from supervision import ByteTrack; t = ByteTrack(); print('ok', type(t).__name__)"
```

Expected: prints `ok ByteTrack`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add supervision (ByteTrack) and torchvision deps"
```

---

## Task 2: ByteTrack adapter

**Files:**
- Create: `src/card_capture/tracking/__init__.py`
- Create: `src/card_capture/tracking/bytetrack_adapter.py`
- Test: `tests/test_bytetrack_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bytetrack_adapter.py
import numpy as np

from card_capture.selector import ScoredCandidate
from card_capture.models import QualityScore
from card_capture.tracking import ByteTrackAdapter


def _candidate(detection_id, frame_index, x, y, conf=0.9, w=200, h=300):
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return ScoredCandidate(
        detection_id=detection_id,
        timestamp_ms=frame_index * 33,
        image_path="x.jpg",
        score=QualityScore(total=conf, sharpness=conf, blur=0.0, area=0.5),
        corners=corners,
        frame_index=frame_index,
    )


def test_adapter_assigns_consistent_track_id_for_overlapping_boxes():
    adapter = ByteTrackAdapter()
    out = []
    for i in range(5):
        cand = _candidate(detection_id=i, frame_index=i, x=100, y=100)
        out.append(adapter.process([cand]))

    track_ids = [a[0].track_id for a in out if a]
    assert len(set(track_ids)) == 1, f"expected single track, got {track_ids}"


def test_adapter_starts_new_track_for_distant_box():
    adapter = ByteTrackAdapter()
    a = _candidate(0, 0, x=100, y=100)
    b = _candidate(1, 1, x=1500, y=1500)  # very far away
    adapter.process([a])
    out = adapter.process([b])
    # b should be its own track, not the same as a's
    assert out[0].track_id != 1 or len(adapter.finalized_tracks()) >= 1


def test_adapter_finalize_returns_track_states():
    adapter = ByteTrackAdapter(min_track_length=2)
    for i in range(3):
        adapter.process([_candidate(i, i, x=100, y=100)])
    tracks = adapter.finalize()
    assert len(tracks) >= 1
    track = tracks[0]
    assert hasattr(track, "candidates")
    assert hasattr(track, "instance_id")
    assert len(track.candidates) >= 2
```

(The `QualityScore` constructor call assumes the existing dataclass shape — verify by `grep -n "class QualityScore" src/card_capture/models.py` and adjust kwargs if needed.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bytetrack_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'card_capture.tracking'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_capture/tracking/__init__.py
from .bytetrack_adapter import ByteTrackAdapter

__all__ = ["ByteTrackAdapter"]
```

```python
# src/card_capture/tracking/bytetrack_adapter.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..selector import ScoredCandidate, TrackState


def _xyxy_from_corners(corners) -> np.ndarray:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)


@dataclass
class _AdaptedDetection:
    candidate: ScoredCandidate
    track_id: int
    instance_id: str


class ByteTrackAdapter:
    """Wraps supervision.ByteTrack to consume ScoredCandidate streams.

    The adapter maintains a stable instance_id (UUID string) per ByteTrack track_id
    so downstream pipeline code keeps its existing identifier shape.
    """

    def __init__(
        self,
        min_track_length: int = 3,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ):
        from supervision import ByteTrack, Detections

        self._ByteTrack = ByteTrack
        self._Detections = Detections
        self._tracker = ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        self.min_track_length = min_track_length
        self._tracks: dict[int, TrackState] = {}  # track_id -> TrackState
        self._all_finalized: list[TrackState] = []

    def reset(self) -> None:
        """Reset tracker state (e.g., between sessions)."""
        from supervision import ByteTrack
        self._all_finalized.extend(self._tracks.values())
        self._tracks = {}
        self._tracker = self._ByteTrack()

    def finalized_tracks(self) -> List[TrackState]:
        return list(self._all_finalized)

    def process(self, candidates: List[ScoredCandidate]) -> List[_AdaptedDetection]:
        """Process detections from one frame; returns adapted detections with track_id."""
        if not candidates:
            return []

        # Build supervision.Detections
        boxes = []
        confidences = []
        for cand in candidates:
            if not cand.corners:
                continue
            boxes.append(_xyxy_from_corners(cand.corners))
            confidences.append(float(cand.score.total))
        if not boxes:
            return []

        det = self._Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
        )
        tracked = self._tracker.update_with_detections(det)

        out: List[_AdaptedDetection] = []
        for i, track_id in enumerate(tracked.tracker_id):
            if track_id is None:
                continue
            tid = int(track_id)
            cand = candidates[i]
            if tid not in self._tracks:
                self._tracks[tid] = TrackState(
                    instance_id=str(uuid.uuid4()),
                    candidates=[],
                    last_centroid=None,
                    last_frame_index=cand.frame_index,
                )
            state = self._tracks[tid]
            state.candidates.append(cand)
            state.last_frame_index = cand.frame_index
            out.append(_AdaptedDetection(candidate=cand, track_id=tid, instance_id=state.instance_id))
        return out

    def finalize(self) -> List[TrackState]:
        """Return all tracks (current + previously reset) above min length."""
        all_tracks = list(self._tracks.values()) + list(self._all_finalized)
        return [t for t in all_tracks if len(t.candidates) >= self.min_track_length]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bytetrack_adapter.py -v`
Expected: 3 PASSED.

If `test_adapter_starts_new_track_for_distant_box` fails because ByteTrack reuses the ID after a single frame, increase the gap by inserting empty `process([])` calls between A and B and re-test.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/tracking/ tests/test_bytetrack_adapter.py
git commit -m "feat(tracking): add ByteTrack adapter consuming ScoredCandidate stream"
```

---

## Task 3: Wire ByteTrack into pipeline.py

**Files:**
- Modify: `src/card_capture/pipeline.py`

- [ ] **Step 1: Replace the import (line 33)**

Find the line:

```python
from .selector import CandidateSelector, HysteresisTracker, ScoredCandidate
```

Change to:

```python
from .selector import CandidateSelector, ScoredCandidate
from .tracking import ByteTrackAdapter
```

- [ ] **Step 2: Replace tracker instantiation (line 169 area)**

Find:

```python
self.tracker = HysteresisTracker(max_dist=150.0, min_track_length=12, max_gap_frames=15)
```

Change to:

```python
self.tracker = ByteTrackAdapter(min_track_length=12)
```

- [ ] **Step 3: Replace the second instantiation (line 230 area)**

Find:

```python
self.tracker = HysteresisTracker(
    ...
    max_gap_frames=options.null_patience_frames * 2,
)
```

Change to:

```python
self.tracker = ByteTrackAdapter(
    min_track_length=options.min_track_length,
    lost_track_buffer=options.null_patience_frames * 2,
)
```

(Verify `options.min_track_length` exists; `grep -n "min_track_length" src/card_capture/pipeline.py` — it's already in `ProcessingOptions`.)

- [ ] **Step 4: Replace per-frame call (line 277 area)**

Find:

```python
self.tracker.tick()
...
self.tracker.process(candidate)
```

The new adapter takes a list of candidates per frame. Change the per-candidate `process(candidate)` call to batch all candidates for the frame:

```python
# Replace this:
#     self.tracker.tick()
#     ...
#     self.tracker.process(candidate)
# With this (collect candidates per frame_index, then call once):
```

Refactor the loop so candidates from the same `frame_index` are batched. Pseudocode:

```python
# Before loop: maintain a buffer of (frame_index, [candidates])
candidates_for_frame: list[ScoredCandidate] = []
current_frame_idx: Optional[int] = None
for cand in stream:
    if current_frame_idx is None:
        current_frame_idx = cand.frame_index
    if cand.frame_index != current_frame_idx:
        self.tracker.process(candidates_for_frame)
        candidates_for_frame = []
        current_frame_idx = cand.frame_index
    candidates_for_frame.append(cand)
if candidates_for_frame:
    self.tracker.process(candidates_for_frame)
```

(Read the current loop carefully — there's surrounding logic for gap detection and resets. Preserve all of it; only change the per-frame tracker call.)

The `tracker.tick()` call is no longer needed (ByteTrack tracks elapsed steps internally). Remove all `self.tracker.tick()` lines.

- [ ] **Step 5: Run pipeline + tests + harness checkpoint**

Run unit tests for areas you didn't touch:

```bash
pytest tests/ -v --ignore=tests/test_hysteresis_tracker.py
```

Expected: green.

Run the harness against the corpus:

```bash
card-capture harness run --baseline reports/baseline_v3.json
```

Expected: report writes; deltas show tracker change effect. Card recall should be no worse than baseline.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/pipeline.py
git commit -m "feat(pipeline): replace HysteresisTracker with ByteTrack adapter"
```

---

## Task 4: Delete HysteresisTracker and its tests

**Files:**
- Modify: `src/card_capture/selector.py`
- Delete: `tests/test_hysteresis_tracker.py`

- [ ] **Step 1: Verify no other consumers**

```bash
grep -rn "HysteresisTracker\|detect_flip" src/ tests/ --include="*.py"
```

Expected: only `selector.py` (definition) and `test_hysteresis_tracker.py` (the test file we're about to delete) appear. If any other file references either name, stop and add a task to migrate it before proceeding.

- [ ] **Step 2: Delete the class**

In `src/card_capture/selector.py`, delete:

- The `HysteresisTracker` class (currently lines ~32–233).
- The `detect_flip` method inside it (already part of the class).

Keep:
- `ScoredCandidate`
- `TrackState`
- `_calculate_centroid`, `_euclidean_distance`, `_get_polygon_area` (still used by `CandidateSelector` and the ByteTrack adapter)
- `CandidateSelector`
- `SpatialCluster`
- `_aspect_ratio`

After editing, run `pytest tests/ --ignore=tests/test_hysteresis_tracker.py -v` — expected green.

- [ ] **Step 3: Delete the test file**

```bash
rm tests/test_hysteresis_tracker.py
```

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/selector.py tests/test_hysteresis_tracker.py
git commit -m "refactor: delete HysteresisTracker (replaced by ByteTrack adapter)"
```

---

## Task 5: Adaptive session gap — compute distribution

**Files:**
- Create: `src/card_capture/adaptive_gap.py`
- Test: `tests/test_adaptive_gap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adaptive_gap.py
import numpy as np

from card_capture.adaptive_gap import compute_session_gap_frames, GapDistribution


def test_compute_gap_returns_p95_plus_buffer():
    # Inter-window gaps in frames: mostly 5, with a few outliers
    gaps = [3, 4, 5, 5, 6, 5, 7, 5, 6, 4, 30]  # P95 around the high tail
    result = compute_session_gap_frames(gaps, fps=30.0)
    assert isinstance(result, GapDistribution)
    assert result.recommended_gap_frames >= 7  # at least P95 of the typical body
    assert result.recommended_gap_frames <= 90  # capped at 3s @ 30fps
    assert result.p50_frames == 5


def test_floor_minimum_when_gaps_tiny():
    gaps = [1, 1, 1, 2, 1]  # all very small
    result = compute_session_gap_frames(gaps, fps=30.0)
    # 0.5s floor at 30fps = 15 frames
    assert result.recommended_gap_frames >= 15


def test_cap_at_three_seconds():
    gaps = [200, 250, 300, 350, 400]  # huge
    result = compute_session_gap_frames(gaps, fps=30.0)
    # 3s cap at 30fps = 90 frames
    assert result.recommended_gap_frames == 90


def test_empty_input_returns_default():
    result = compute_session_gap_frames([], fps=30.0)
    assert result.recommended_gap_frames == 15  # 0.5s default at 30fps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaptive_gap.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_capture/adaptive_gap.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GapDistribution:
    p50_frames: int
    p95_frames: int
    recommended_gap_frames: int


def compute_session_gap_frames(
    inter_window_gaps_frames: Sequence[int],
    *,
    fps: float = 30.0,
    floor_seconds: float = 0.5,
    cap_seconds: float = 3.0,
    safety_pad_frames: int = 2,
) -> GapDistribution:
    floor = int(round(floor_seconds * fps))
    cap = int(round(cap_seconds * fps))

    if not inter_window_gaps_frames:
        return GapDistribution(p50_frames=0, p95_frames=0, recommended_gap_frames=floor)

    arr = np.asarray(list(inter_window_gaps_frames), dtype=np.float32)
    p50 = int(np.percentile(arr, 50))
    p95 = int(np.percentile(arr, 95))
    recommended = max(floor, min(cap, p95 + safety_pad_frames))
    return GapDistribution(p50_frames=p50, p95_frames=p95, recommended_gap_frames=recommended)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adaptive_gap.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/adaptive_gap.py tests/test_adaptive_gap.py
git commit -m "feat(adaptive-gap): compute per-video session gap from gap distribution"
```

---

## Task 6: Wire adaptive gap into pipeline + sampler telemetry

**Files:**
- Modify: `src/card_capture/sampler.py` — expose inter-window gaps from Pass 1
- Modify: `src/card_capture/pipeline.py` — consume them

- [ ] **Step 1: Expose inter-window gaps from `AdaptivePresenceSampler`**

In `src/card_capture/sampler.py`, find the `AdaptivePresenceSampler` class. After `last_score_threshold` (around line 307), add:

```python
        self.last_inter_window_gaps_frames: list[int] = []
```

In `_build_windows`, after `windows` is constructed but before returning, compute the gaps:

```python
        # New: record inter-window gaps for adaptive session-gap computation
        self.last_inter_window_gaps_frames = [
            (windows[i + 1].start_frame - windows[i].end_frame)
            for i in range(len(windows) - 1)
        ]
```

(Apply this to all three return paths: the main path, the empty-windows fallback, and the no-windows return.)

- [ ] **Step 2: Use the gap distribution in `pipeline.py`**

In `src/card_capture/pipeline.py`, after the sampler completes Pass 1 and you have access to `sampler.last_inter_window_gaps_frames`, add (find the `null_patience_frames` use site around line 251):

```python
from .adaptive_gap import compute_session_gap_frames

# After Pass 1, before the main per-frame loop begins:
gap_dist = compute_session_gap_frames(
    sampler.last_inter_window_gaps_frames,
    fps=video_fps,  # already known from existing telemetry
)
effective_session_gap_frames = min(
    options.null_patience_frames,  # max bound from config
    gap_dist.recommended_gap_frames,
)
```

Then change the gap-detection condition:

```python
# Before:
#     if last_frame_idx != -1 and (frame_index - last_frame_idx) > options.null_patience_frames:
# After:
    if last_frame_idx != -1 and (frame_index - last_frame_idx) > effective_session_gap_frames:
```

Where `video_fps` comes from: search for `cv2.CAP_PROP_FPS` in the file. If not stored, capture it once when opening the video and pass it through.

Also write the gap distribution into telemetry so the Timeline UI shows it:

```python
sampler_telemetry["adaptive_gap_p50"] = gap_dist.p50_frames
sampler_telemetry["adaptive_gap_p95"] = gap_dist.p95_frames
sampler_telemetry["adaptive_gap_recommended"] = gap_dist.recommended_gap_frames
sampler_telemetry["adaptive_gap_effective"] = effective_session_gap_frames
```

- [ ] **Step 3: Run harness**

```bash
pytest tests/ -v
card-capture harness run --baseline reports/baseline_v3.json
```

Expected: tests green; harness shows session-boundary changes per-video.

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/sampler.py src/card_capture/pipeline.py
git commit -m "feat(pipeline): use per-video adaptive session gap from Pass 1 distribution"
```

---

## Task 7: Presence training data mining

**Files:**
- Create: `src/card_capture/presence/__init__.py`
- Create: `src/card_capture/presence/training_data.py`
- Test: `tests/test_presence_training_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_presence_training_data.py
import numpy as np
import cv2

from card_capture.presence.training_data import sample_negative_patches, mine_positive_crops


def test_sample_negative_patches_returns_correct_shape():
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    patches = sample_negative_patches(frame, count=4, patch_size=224, rng_seed=42)
    assert len(patches) == 4
    for p in patches:
        assert p.shape == (224, 224, 3)
        assert p.dtype == np.uint8


def test_sample_negative_patches_skips_when_frame_too_small():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    patches = sample_negative_patches(frame, count=4, patch_size=224, rng_seed=42)
    assert patches == []


def test_mine_positive_crops_extracts_card_region():
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    # Simulated detection corners in the center
    corners = [(800, 400), (1100, 400), (1100, 700), (800, 700)]
    crops = mine_positive_crops(frame, [corners], pad_ratio=0.0, target_size=224)
    assert len(crops) == 1
    assert crops[0].shape == (224, 224, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_presence_training_data.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_capture/presence/__init__.py
# (empty marker)
```

```python
# src/card_capture/presence/training_data.py
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def sample_negative_patches(
    frame: np.ndarray,
    count: int,
    patch_size: int = 224,
    rng_seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Sample random patches from a frame as negative training examples."""
    h, w = frame.shape[:2]
    if h < patch_size or w < patch_size:
        return []
    rng = random.Random(rng_seed)
    out: List[np.ndarray] = []
    for _ in range(count):
        y = rng.randint(0, h - patch_size)
        x = rng.randint(0, w - patch_size)
        out.append(frame[y:y + patch_size, x:x + patch_size].copy())
    return out


def mine_positive_crops(
    frame: np.ndarray,
    corners_per_card: Sequence[Sequence[Tuple[float, float]]],
    pad_ratio: float = 0.05,
    target_size: int = 224,
) -> List[np.ndarray]:
    """For each set of card corners in the frame, extract an axis-aligned crop sized for training."""
    h, w = frame.shape[:2]
    out: List[np.ndarray] = []
    for corners in corners_per_card:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        x0 = max(0, int(min(xs) - pad_ratio * (max(xs) - min(xs))))
        x1 = min(w, int(max(xs) + pad_ratio * (max(xs) - min(xs))))
        y0 = max(0, int(min(ys) - pad_ratio * (max(ys) - min(ys))))
        y1 = min(h, int(max(ys) + pad_ratio * (max(ys) - min(ys))))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = frame[y0:y1, x0:x1]
        resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        out.append(resized)
    return out


def export_dataset(
    db_path: Path,
    video_id: int,
    out_dir: Path,
    *,
    confidence_floor: float = 0.7,
    negatives_per_frame: int = 2,
    target_size: int = 224,
) -> Tuple[int, int]:
    """Iterate over a video's frames, write positives + negatives to disk.

    Returns (positive_count, negative_count).
    """
    from card_capture.storage import Storage
    storage = Storage(db_path)
    storage.initialize()
    out_pos = out_dir / "positives"
    out_neg = out_dir / "negatives"
    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)

    pos_n = 0
    neg_n = 0
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT cv.id, cv.frame_index, cv.timestamp_ms, cv.image_path,
                   cv.corners_json, cv.confidence
            FROM card_views cv
            JOIN card_instances ci ON ci.id = cv.card_instance_id
            WHERE ci.video_id = ? AND cv.confidence >= ?
            ORDER BY cv.frame_index
            """,
            (video_id, confidence_floor),
        ).fetchall()

    import json as _json
    rng_seed = 0
    for row in rows:
        frame = cv2.imread(row["image_path"])
        if frame is None:
            continue
        corners_list = _json.loads(row["corners_json"]) if row["corners_json"] else []
        if not corners_list:
            continue
        corners_per_card = [corners_list] if isinstance(corners_list[0][0], (int, float)) else corners_list
        for crop in mine_positive_crops(frame, corners_per_card, target_size=target_size):
            cv2.imwrite(str(out_pos / f"v{video_id}_f{row['frame_index']}_p{pos_n}.jpg"), crop)
            pos_n += 1
        for patch in sample_negative_patches(frame, count=negatives_per_frame, patch_size=target_size, rng_seed=rng_seed):
            cv2.imwrite(str(out_neg / f"v{video_id}_f{row['frame_index']}_n{neg_n}.jpg"), patch)
            neg_n += 1
        rng_seed += 1

    return pos_n, neg_n
```

(The exact column names — `corners_json`, `image_path`, `confidence` — must be verified against `src/card_capture/storage.py`. If different, adjust the SELECT and the row access. Run `grep -n "CREATE TABLE card_views" src/card_capture/storage.py` before relying on these.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_presence_training_data.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/presence/ tests/test_presence_training_data.py
git commit -m "feat(presence): add training-data mining (positives + negative patches)"
```

---

## Task 8: Generate the dataset

**Files:**
- Create: `data/presence_dataset/positives/` and `data/presence_dataset/negatives/` (gitignored)

- [ ] **Step 1: Add data dir to gitignore**

Append to `.gitignore`:

```
data/
```

- [ ] **Step 2: Run mining for each labeled video**

For each video_id present in `card_capture_output/cards.sqlite`:

```bash
python -c "
from pathlib import Path
from card_capture.presence.training_data import export_dataset
pos, neg = export_dataset(
    db_path=Path('card_capture_output/cards.sqlite'),
    video_id=VIDEO_ID,
    out_dir=Path('data/presence_dataset'),
)
print(f'video VIDEO_ID: {pos} positives, {neg} negatives')
"
```

Replace `VIDEO_ID` with each integer video id (find with `sqlite3 card_capture_output/cards.sqlite "SELECT id FROM videos"`).

Expected: total ~500–2000 positives and a similar number of negatives. If totals are below 200 positives, add a step to manually sample more positives from the labeling-mode UI.

- [ ] **Step 3: Sanity-check the dataset visually**

```bash
ls data/presence_dataset/positives/ | wc -l
ls data/presence_dataset/negatives/ | wc -l
open data/presence_dataset/positives/$(ls data/presence_dataset/positives/ | head -1)
```

Expected: opens a card-shaped 224x224 image. Verify a few negatives don't accidentally contain cards.

- [ ] **Step 4: Commit gitignore only**

```bash
git add .gitignore
git commit -m "chore: gitignore generated training data dir"
```

---

## Task 9: Train MobileNetV3-Small presence classifier

**Files:**
- Create: `src/card_capture/train/__init__.py`
- Create: `src/card_capture/train/presence.py`
- Create: `models/.gitkeep`

- [ ] **Step 1: Write the training script**

```python
# src/card_capture/train/__init__.py
# (empty marker)
```

```python
# src/card_capture/train/presence.py
"""Train MobileNetV3-Small presence classifier.

Usage:
    python -m card_capture.train.presence \
        --data data/presence_dataset --out models/presence_classifier.pt \
        --epochs 8 --batch-size 64
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
import cv2
import numpy as np


class _PresenceDataset(Dataset):
    def __init__(self, root: Path, train: bool):
        self.samples: list[Tuple[Path, int]] = []
        for label, sub in enumerate(["negatives", "positives"]):  # negative=0, positive=1
            for path in sorted((root / sub).glob("*.jpg")):
                self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"no samples under {root}")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            *([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ] if train else []),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"could not read {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), label


def _build_model() -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    device = _device()
    print(f"device: {device}")

    ds = _PresenceDataset(args.data, train=True)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = _build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device); y = y.to(device)
                preds = model(x).argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += int(y.size(0))
        acc = correct / total if total else 0.0
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "val_acc": acc}, args.out)
            print(f"  saved {args.out} (val_acc={acc:.4f})")

    print(f"best val_acc={best_acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Set up models directory**

```bash
mkdir -p models
touch models/.gitkeep
```

Append to `.gitignore`:

```
models/*
!models/.gitkeep
!models/presence_classifier.pt
```

- [ ] **Step 3: Train**

```bash
python -m card_capture.train.presence \
  --data data/presence_dataset \
  --out models/presence_classifier.pt \
  --epochs 8 --batch-size 64
```

Expected: prints per-epoch train_loss + val_acc; final val_acc ≥ 0.95. Final file `models/presence_classifier.pt` exists.

If val_acc < 0.90 after 8 epochs:
- Check class balance: `ls data/presence_dataset/positives | wc -l` vs `ls data/presence_dataset/negatives | wc -l`. If imbalanced > 3:1, mine more of the minority class.
- Increase `--epochs` to 16.
- If still bad, capture sample positives + negatives and inspect manually for label noise.

- [ ] **Step 4: Commit the trained model**

```bash
git add models/.gitkeep .gitignore models/presence_classifier.pt src/card_capture/train/
git commit -m "feat(presence): train MobileNetV3-Small classifier (val_acc TBD)"
```

(Replace `TBD` in the commit message with the actual val_acc printed.)

---

## Task 10: Presence classifier inference wrapper

**Files:**
- Create: `src/card_capture/presence/classifier.py`
- Test: `tests/test_presence_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_presence_classifier.py
import os
from pathlib import Path

import numpy as np
import pytest

from card_capture.presence.classifier import PresenceClassifier

WEIGHTS = Path("models/presence_classifier.pt")
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="weights not trained yet")


def test_classifier_returns_score_in_unit_interval():
    clf = PresenceClassifier(weights_path=WEIGHTS)
    frame = np.full((300, 300, 3), 200, dtype=np.uint8)
    score = clf.score(frame)
    assert 0.0 <= score <= 1.0


def test_classifier_batch_returns_list():
    clf = PresenceClassifier(weights_path=WEIGHTS)
    frames = [np.full((300, 300, 3), v, dtype=np.uint8) for v in (50, 100, 150, 200)]
    scores = clf.score_batch(frames)
    assert len(scores) == 4
    for s in scores:
        assert 0.0 <= s <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_presence_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError` (or skipped if weights missing — train them in Task 9 first).

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_capture/presence/classifier.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


def _build_model() -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model


def _resolve_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PresenceClassifier:
    """Tiny binary classifier: returns P(card present) for an input frame/patch."""

    def __init__(self, weights_path: Path, device: str = "auto"):
        self.device = _resolve_device(device)
        self.model = _build_model().to(self.device)
        ckpt = torch.load(str(weights_path), map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.tx = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _prep(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.tx(rgb)

    def score(self, frame_bgr: np.ndarray) -> float:
        with torch.no_grad():
            x = self._prep(frame_bgr).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].item())  # P(class=1=positive)

    def score_batch(self, frames_bgr: List[np.ndarray]) -> List[float]:
        if not frames_bgr:
            return []
        with torch.no_grad():
            xs = torch.stack([self._prep(f) for f in frames_bgr]).to(self.device)
            logits = self.model(xs)
            probs = torch.softmax(logits, dim=1)
            return probs[:, 1].cpu().tolist()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_presence_classifier.py -v`
Expected: 2 PASSED (assuming Task 9 produced the weights).

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/presence/classifier.py tests/test_presence_classifier.py
git commit -m "feat(presence): add inference wrapper (MPS/CUDA/CPU)"
```

---

## Task 11: Wire presence classifier into AdaptivePresenceSampler

**Files:**
- Modify: `src/card_capture/sampler.py`

- [ ] **Step 1: Replace Otsu thresholding with classifier output**

In `src/card_capture/sampler.py`, find `AdaptivePresenceSampler._build_windows` (around line 461). Currently it calls `_otsu_threshold(scores)` and produces `active_flags`.

Add a new constructor parameter `presence_classifier: Optional[PresenceClassifier] = None`. Store it on `self`.

Replace the `active_flags` block with:

```python
        if self.presence_classifier is not None:
            # Use the visual classifier on each scan-resolution proxy frame.
            # Chunk to bound peak memory (sampler can hold thousands of frames).
            scores: list[float] = []
            chunk_size = 32
            for start in range(0, len(records), chunk_size):
                chunk_frames = [r.image for r in records[start:start + chunk_size]]
                scores.extend(self.presence_classifier.score_batch(chunk_frames))
            for record, score in zip(records, scores):
                record.presence_score = score
            active_flags = [s >= 0.5 for s in scores]
        else:
            # Fallback: existing composite z-score path
            scores = self._score_records(records)
            for record, score in zip(records, scores):
                record.presence_score = score
            threshold = self._otsu_threshold(scores)
            edge_vals = np.array([r.metrics["edge_density"] for r in records])
            edge_median = float(np.median(edge_vals))
            edge_mad = float(np.median(np.abs(edge_vals - edge_median)))
            edge_threshold = edge_median + (2.5 * edge_mad * 1.4826) if edge_mad > 1e-6 else float('inf')
            active_flags = []
            for idx, score in enumerate(scores):
                is_otsu_active = score > threshold
                is_feature_active = records[idx].metrics["edge_density"] > edge_threshold
                active_flags.append(is_otsu_active or is_feature_active)
```

Add the import at the top of sampler.py:

```python
from .presence.classifier import PresenceClassifier
```

- [ ] **Step 2: Wire the classifier in `cli.py`**

In `src/card_capture/cli.py`, where `AdaptivePresenceSampler(...)` is constructed (around line 67), add:

```python
        from card_capture.presence.classifier import PresenceClassifier
        from pathlib import Path as _P
        weights = _P("models/presence_classifier.pt")
        presence_clf = PresenceClassifier(weights_path=weights, device=config.device) if weights.exists() else None
        sampler = AdaptivePresenceSampler(
            video_path=args.video_path,
            reader_backend=config.reader_backend,
            device=config.device,
            presence_classifier=presence_clf,
        )
```

- [ ] **Step 3: Run unit tests + harness**

```bash
pytest tests/ -v
card-capture harness run --baseline reports/baseline_v3.json
```

Expected:
- Unit tests green.
- Harness phantom rate decreases (since the classifier rejects hands/packaging that pixel stats let through).
- Card recall does not regress meaningfully.

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/sampler.py src/card_capture/cli.py
git commit -m "feat(sampler): use MobileNetV3-Small classifier for Stage-1 presence"
```

---

## Task 12: Phase A gate check

**Files:**
- Create: `reports/phase_a.json`, `reports/phase_a.md` (or whatever sha emits)

- [ ] **Step 1: Run the harness against the corpus**

```bash
card-capture harness run --baseline reports/baseline_v3.json
```

- [ ] **Step 2: Inspect deltas vs gate**

Open the produced `reports/<sha>.md`. Verify against the soft gate from the spec:

- Card recall ≥ baseline + 20 pp **OR** ≥ 0.95 absolute (whichever is looser).
- Phantom rate ≤ baseline / 2 **OR** ≤ 2% absolute (whichever is looser).
- Wall-clock not regressed by > 30%.

- [ ] **Step 3: Save the Phase A report under a stable name**

```bash
cp reports/<sha>.json reports/phase_a.json
cp reports/<sha>.md reports/phase_a.md
git add reports/phase_a.json reports/phase_a.md
git commit -m "docs(harness): capture Phase A report (tracker + presence classifier + adaptive gap)"
```

- [ ] **Step 4: Phase A complete — report results to user**

Stop here. Surface the harness deltas to the user:

- Recall before/after.
- Phantom rate before/after.
- Wall-clock before/after.
- ID switches per video.

Pause for user direction on whether to proceed to Phase B planning.

---

## Self-review

- All steps include actual code or actual commands; no placeholders except for one `TBD` in a commit message that the engineer fills in from terminal output.
- File paths exact throughout.
- Tests precede implementation.
- Commits at end of every task.
- Spec coverage:
  - §A1 presence classifier ✓ (Tasks 7, 8, 9, 10, 11).
  - §A2 ByteTrack tracker ✓ (Tasks 1, 2, 3, 4).
  - §A3 adaptive session gap ✓ (Tasks 5, 6).
  - §A4 delete area-drop rule ✓ (Task 4 — class deletion includes `detect_flip`).
  - §Phase A deliverables (presence weights, training script, ByteTrack adapter, sampler updates, harness report) ✓.
  - §Phase A gate ✓ (Task 12).
- Type consistency: `ByteTrackAdapter` exposes `process()`, `finalize()`, `reset()`, `finalized_tracks()` — matches the call sites changed in Task 3. `TrackState` reused (defined in `selector.py`, kept after deletion in Task 4). `PresenceClassifier` exposes `score()` and `score_batch()` — matches Task 11 wiring.
- Open issues called out inline: `phash`/`corners_json` column names need verification in storage.py (Tasks 2, 7); `min_track_length` field on `ProcessingOptions` (Task 3); `video_fps` plumbing (Task 6).
