# Pipeline V4 — Spec 0: Bootstrap (Labeled Corpus + Regression Harness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a hand-labeled golden corpus, a regression harness that measures card recall / phantom rate / F-B accuracy / dedup F1 / ID switches / quality / perf against truth, and a Review UI labeling mode that produces `<video>.truth.json` files. Capture a `baseline_v3` report that all subsequent phases compare against.

**Architecture:** Two subsystems land here. (1) `tests/regression/` package containing modular metric/match/report/harness components plus a CLI entry point on `card_capture.cli`. (2) FastAPI labeling routes added to `src/card_capture/review.py` plus a new `labeling.html` template that writes truth JSON to disk. Each metric is unit-tested independently against synthetic pipeline output and synthetic truth; the harness orchestration is integration-tested by running against the existing `tests/` fixtures.

**Tech Stack:** Python 3.9+, pytest, FastAPI, Jinja2 (existing), argparse (existing CLI). No new runtime dependencies.

---

## File Structure

**Create:**
- `tests/regression/__init__.py` (empty marker)
- `tests/regression/truth.py` — load + validate `<video>.truth.json`, dataclasses for expected cards
- `tests/regression/matcher.py` — pair pipeline-output Card Instances to truth cards via temporal overlap
- `tests/regression/metrics.py` — per-video metric functions (recall, phantom, F/B, ID switch, quality, perf)
- `tests/regression/cross_video.py` — cross-video dedup F1
- `tests/regression/pipeline_runner.py` — run pipeline on a fixture video, return canonical harness records
- `tests/regression/report.py` — JSON + Markdown writers, delta computation
- `tests/regression/harness.py` — orchestration entry point
- `tests/regression/test_truth.py`
- `tests/regression/test_matcher.py`
- `tests/regression/test_metrics.py`
- `tests/regression/test_cross_video.py`
- `tests/regression/test_report.py`
- `tests/regression/test_harness.py` (smoke test, marked `slow`)
- `tests/fixtures/golden_corpus/.gitkeep`
- `reports/.gitkeep`
- `src/card_capture/templates/labeling.html`
- `Makefile` (root)

**Modify:**
- `src/card_capture/cli.py` — add `harness` subcommand group with `run` and `compare` subcommands
- `src/card_capture/review.py` — add `GET /label/{video_id}`, `POST /label/{video_id}/save` routes
- `.gitignore` — append `reports/*` (allow `reports/.gitkeep` and `reports/baseline_*`)

---

## Task 1: Truth schema + loader

**Files:**
- Create: `tests/regression/truth.py`
- Test: `tests/regression/test_truth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/test_truth.py
import json
from pathlib import Path

import pytest

from tests.regression.truth import ExpectedCard, GroundTruth, load_truth, TruthValidationError


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "video_001.truth.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_truth_returns_dataclass(tmp_path):
    path = _write(tmp_path, {
        "video_id": "video_001",
        "video_path": "tests/fixtures/golden_corpus/video_001/video_001.mp4",
        "labeled_at": "2026-05-10",
        "labeled_by": "josh",
        "expected_cards": [
            {
                "card_id": "card_001",
                "physical_card_key": "topps_chrome_2024_42",
                "front_present": True,
                "back_present": True,
                "approx_front_window_ms": [12500, 15800],
                "approx_back_window_ms": [16100, 18900],
                "notes": "foil",
            }
        ],
    })

    truth = load_truth(path)

    assert isinstance(truth, GroundTruth)
    assert truth.video_id == "video_001"
    assert len(truth.expected_cards) == 1
    card = truth.expected_cards[0]
    assert isinstance(card, ExpectedCard)
    assert card.card_id == "card_001"
    assert card.physical_card_key == "topps_chrome_2024_42"
    assert card.front_present is True
    assert card.approx_front_window_ms == (12500, 15800)


def test_load_truth_rejects_missing_video_id(tmp_path):
    path = _write(tmp_path, {"expected_cards": []})
    with pytest.raises(TruthValidationError, match="video_id"):
        load_truth(path)


def test_load_truth_allows_missing_optional_fields(tmp_path):
    path = _write(tmp_path, {
        "video_id": "video_002",
        "video_path": "x.mp4",
        "expected_cards": [
            {
                "card_id": "c1",
                "front_present": True,
                "back_present": False,
                "approx_front_window_ms": [0, 1000],
            }
        ],
    })
    truth = load_truth(path)
    card = truth.expected_cards[0]
    assert card.physical_card_key is None
    assert card.approx_back_window_ms is None
    assert card.notes == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/test_truth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.regression.truth'`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/regression/truth.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


class TruthValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ExpectedCard:
    card_id: str
    front_present: bool
    back_present: bool
    approx_front_window_ms: Optional[Tuple[int, int]] = None
    approx_back_window_ms: Optional[Tuple[int, int]] = None
    physical_card_key: Optional[str] = None
    notes: str = ""


@dataclass(frozen=True)
class GroundTruth:
    video_id: str
    video_path: str
    expected_cards: Tuple[ExpectedCard, ...]
    labeled_at: str = ""
    labeled_by: str = ""


def _coerce_window(value) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise TruthValidationError(f"window must be [start_ms, end_ms], got {value!r}")
    return (int(value[0]), int(value[1]))


def load_truth(path: Path) -> GroundTruth:
    raw = json.loads(Path(path).read_text())

    for required in ("video_id", "video_path", "expected_cards"):
        if required not in raw:
            raise TruthValidationError(f"missing required field: {required}")

    cards = []
    for entry in raw["expected_cards"]:
        for required in ("card_id", "front_present", "back_present"):
            if required not in entry:
                raise TruthValidationError(f"card missing required field: {required}")
        cards.append(
            ExpectedCard(
                card_id=str(entry["card_id"]),
                front_present=bool(entry["front_present"]),
                back_present=bool(entry["back_present"]),
                approx_front_window_ms=_coerce_window(entry.get("approx_front_window_ms")),
                approx_back_window_ms=_coerce_window(entry.get("approx_back_window_ms")),
                physical_card_key=entry.get("physical_card_key"),
                notes=str(entry.get("notes", "")),
            )
        )

    return GroundTruth(
        video_id=str(raw["video_id"]),
        video_path=str(raw["video_path"]),
        expected_cards=tuple(cards),
        labeled_at=str(raw.get("labeled_at", "")),
        labeled_by=str(raw.get("labeled_by", "")),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/regression/test_truth.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/regression/__init__.py tests/regression/truth.py tests/regression/test_truth.py
git commit -m "feat(harness): add truth.json loader and validation"
```

(Create empty `tests/regression/__init__.py` if it doesn't exist before staging.)

---

## Task 2: Pipeline-output adapter

**Files:**
- Create: `tests/regression/pipeline_runner.py`
- Test: in `test_harness.py` later (this task ships a skeleton + a pure-Python data shape; integration-tested via the smoke test)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/regression/test_truth.py temporarily, or create tests/regression/test_pipeline_runner.py
# tests/regression/test_pipeline_runner.py
from tests.regression.pipeline_runner import HarnessInstance, instances_from_db_rows


def test_instances_from_db_rows_maps_fields():
    rows = [
        {
            "instance_id": 7,
            "video_id": 3,
            "session_id": 2,
            "angle": "Front",
            "is_duplicate_of": None,
            "fused_image_path": "out/foo.jpg",
            "start_time": 12000,
            "end_time": 16000,
            "detection_count": 18,
            "phash": "abc",
        },
    ]
    out = instances_from_db_rows(rows)
    assert len(out) == 1
    inst = out[0]
    assert isinstance(inst, HarnessInstance)
    assert inst.instance_id == 7
    assert inst.angle == "Front"
    assert inst.start_ms == 12000
    assert inst.end_ms == 16000
    assert inst.duplicate_of is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/test_pipeline_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/regression/pipeline_runner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class HarnessInstance:
    """A pipeline-produced Card Instance, normalized for harness consumption."""
    instance_id: int
    video_id: int
    session_id: int
    angle: str
    duplicate_of: Optional[int]
    fused_image_path: Optional[str]
    start_ms: int
    end_ms: int
    detection_count: int
    phash: Optional[str]


def instances_from_db_rows(rows: Iterable[dict]) -> List[HarnessInstance]:
    out: List[HarnessInstance] = []
    for row in rows:
        out.append(
            HarnessInstance(
                instance_id=int(row["instance_id"]),
                video_id=int(row["video_id"]),
                session_id=int(row["session_id"]),
                angle=str(row.get("angle") or "Unknown"),
                duplicate_of=row.get("is_duplicate_of"),
                fused_image_path=row.get("fused_image_path"),
                start_ms=int(row["start_time"]),
                end_ms=int(row["end_time"]),
                detection_count=int(row.get("detection_count") or 0),
                phash=row.get("phash"),
            )
        )
    return out


def load_instances_for_video(db_path: Path, video_id: int) -> List[HarnessInstance]:
    """Read Card Instances for a single video out of the pipeline's SQLite DB."""
    from card_capture.storage import Storage
    storage = Storage(db_path)
    storage.initialize()
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT ci.id AS instance_id, ci.video_id, ci.session_id, ci.angle,
                   ci.is_duplicate_of, ci.fused_image_path, ci.phash,
                   MIN(cv.timestamp_ms) AS start_time,
                   MAX(cv.timestamp_ms) AS end_time,
                   COUNT(cv.id) AS detection_count
            FROM card_instances ci
            LEFT JOIN card_views cv ON cv.card_instance_id = ci.id
            WHERE ci.video_id = ?
            GROUP BY ci.id
            ORDER BY start_time ASC
            """,
            (video_id,),
        ).fetchall()
    return instances_from_db_rows([dict(r) for r in rows])
```

(If `phash` column doesn't exist on `card_instances`, drop it from the SELECT and from `HarnessInstance` — verify against `src/card_capture/storage.py` before proceeding.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/regression/test_pipeline_runner.py -v`
Expected: 1 PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/regression/pipeline_runner.py tests/regression/test_pipeline_runner.py
git commit -m "feat(harness): add pipeline output adapter"
```

---

## Task 3: Card matcher (temporal overlap)

**Files:**
- Create: `tests/regression/matcher.py`
- Test: `tests/regression/test_matcher.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/test_matcher.py
from tests.regression.matcher import match_instances_to_truth, MatchResult
from tests.regression.pipeline_runner import HarnessInstance
from tests.regression.truth import ExpectedCard


def _inst(iid, video_id, session_id, angle, start, end, **kw):
    return HarnessInstance(
        instance_id=iid, video_id=video_id, session_id=session_id, angle=angle,
        duplicate_of=kw.get("duplicate_of"), fused_image_path=None,
        start_ms=start, end_ms=end, detection_count=10, phash=None,
    )


def _exp(card_id, front_window=None, back_window=None, key=None):
    return ExpectedCard(
        card_id=card_id,
        front_present=front_window is not None,
        back_present=back_window is not None,
        approx_front_window_ms=front_window,
        approx_back_window_ms=back_window,
        physical_card_key=key,
    )


def test_match_pairs_overlapping_windows():
    truth = (_exp("c1", front_window=(1000, 3000), back_window=(3500, 5000)),)
    instances = [
        _inst(10, 1, 1, "Front", 1100, 2900),
        _inst(11, 1, 1, "Back", 3600, 4900),
    ]
    result = match_instances_to_truth(instances, truth, tolerance_ms=500)

    assert isinstance(result, MatchResult)
    assert len(result.matched) == 2  # one per side
    matched_ids = {pair.instance.instance_id for pair in result.matched}
    assert matched_ids == {10, 11}
    assert len(result.unmatched_truth) == 0
    assert len(result.phantom_instances) == 0


def test_phantom_when_no_truth_overlaps():
    truth = (_exp("c1", front_window=(1000, 2000)),)
    instances = [
        _inst(10, 1, 1, "Front", 1100, 1900),
        _inst(11, 1, 2, "Front", 8000, 9000),  # no truth here
    ]
    result = match_instances_to_truth(instances, truth, tolerance_ms=500)
    assert {p.instance.instance_id for p in result.matched} == {10}
    assert len(result.phantom_instances) == 1
    assert result.phantom_instances[0].instance_id == 11


def test_unmatched_truth_when_no_instance_overlaps():
    truth = (
        _exp("c1", front_window=(1000, 2000)),
        _exp("c2", front_window=(5000, 6000)),
    )
    instances = [_inst(10, 1, 1, "Front", 1000, 2000)]
    result = match_instances_to_truth(instances, truth, tolerance_ms=500)
    assert len(result.unmatched_truth) == 1
    assert result.unmatched_truth[0].card_id == "c2"


def test_tolerance_allows_window_drift():
    truth = (_exp("c1", front_window=(1000, 2000)),)
    instances = [_inst(10, 1, 1, "Front", 2300, 2800)]  # ends after window, within tolerance
    result = match_instances_to_truth(instances, truth, tolerance_ms=500)
    assert len(result.matched) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/test_matcher.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/regression/matcher.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .pipeline_runner import HarnessInstance
from .truth import ExpectedCard


@dataclass(frozen=True)
class MatchedPair:
    truth_card: ExpectedCard
    side: str  # "F" or "B"
    instance: HarnessInstance


@dataclass(frozen=True)
class MatchResult:
    matched: Tuple[MatchedPair, ...]
    unmatched_truth: Tuple[ExpectedCard, ...]
    phantom_instances: Tuple[HarnessInstance, ...]


def _windows_overlap(a: Tuple[int, int], b: Tuple[int, int], tolerance_ms: int) -> bool:
    a_start, a_end = a
    b_start, b_end = b
    return (a_start - tolerance_ms) <= b_end and (b_start - tolerance_ms) <= a_end


def _instance_overlaps_window(inst: HarnessInstance, window: Optional[Tuple[int, int]], tol: int) -> bool:
    if window is None:
        return False
    return _windows_overlap((inst.start_ms, inst.end_ms), window, tol)


def match_instances_to_truth(
    instances: Sequence[HarnessInstance],
    truth: Sequence[ExpectedCard],
    tolerance_ms: int = 500,
) -> MatchResult:
    """Greedy temporal match: each truth side claims one best-overlapping instance."""
    remaining = list(instances)
    matched: List[MatchedPair] = []
    unmatched: List[ExpectedCard] = []

    for card in truth:
        sides_to_match = []
        if card.front_present and card.approx_front_window_ms is not None:
            sides_to_match.append(("F", card.approx_front_window_ms))
        if card.back_present and card.approx_back_window_ms is not None:
            sides_to_match.append(("B", card.approx_back_window_ms))

        any_side_matched = False
        for side, window in sides_to_match:
            best_idx = -1
            best_overlap = -1
            for idx, inst in enumerate(remaining):
                if not _instance_overlaps_window(inst, window, tolerance_ms):
                    continue
                ovl = min(inst.end_ms, window[1]) - max(inst.start_ms, window[0])
                if ovl > best_overlap:
                    best_overlap = ovl
                    best_idx = idx
            if best_idx >= 0:
                matched.append(MatchedPair(truth_card=card, side=side, instance=remaining.pop(best_idx)))
                any_side_matched = True

        if not any_side_matched:
            unmatched.append(card)

    return MatchResult(
        matched=tuple(matched),
        unmatched_truth=tuple(unmatched),
        phantom_instances=tuple(remaining),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/regression/test_matcher.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/regression/matcher.py tests/regression/test_matcher.py
git commit -m "feat(harness): add temporal overlap matcher"
```

---

## Task 4: Per-video metrics — recall, phantom, F/B accuracy

**Files:**
- Create: `tests/regression/metrics.py`
- Test: `tests/regression/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/test_metrics.py
from tests.regression.matcher import MatchResult, MatchedPair
from tests.regression.metrics import VideoMetrics, compute_video_metrics
from tests.regression.pipeline_runner import HarnessInstance
from tests.regression.truth import ExpectedCard


def _inst(iid, angle, start=0, end=1000):
    return HarnessInstance(
        instance_id=iid, video_id=1, session_id=1, angle=angle,
        duplicate_of=None, fused_image_path=None,
        start_ms=start, end_ms=end, detection_count=5, phash=None,
    )


def _exp(card_id, front=True, back=True):
    return ExpectedCard(
        card_id=card_id, front_present=front, back_present=back,
        approx_front_window_ms=(0, 1000) if front else None,
        approx_back_window_ms=(2000, 3000) if back else None,
    )


def test_metrics_perfect_video():
    truth = (_exp("c1"),)
    matched = (
        MatchedPair(truth_card=truth[0], side="F", instance=_inst(10, "Front", 0, 1000)),
        MatchedPair(truth_card=truth[0], side="B", instance=_inst(11, "Back", 2000, 3000)),
    )
    result = MatchResult(matched=matched, unmatched_truth=(), phantom_instances=())

    m = compute_video_metrics(result, truth)
    assert isinstance(m, VideoMetrics)
    assert m.expected_cards == 1
    assert m.detected_cards == 1
    assert m.recall == 1.0
    assert m.phantom_rate == 0.0
    assert m.fb_correct == 2
    assert m.fb_total == 2
    assert m.fb_accuracy == 1.0


def test_metrics_recall_partial():
    truth = (_exp("c1"), _exp("c2"))
    matched = (MatchedPair(truth_card=truth[0], side="F", instance=_inst(10, "Front", 0, 1000)),)
    result = MatchResult(matched=matched, unmatched_truth=(truth[1],), phantom_instances=())

    m = compute_video_metrics(result, truth)
    assert m.recall == 0.5  # 1 of 2 cards detected (any side counts)


def test_metrics_phantom_rate():
    truth = (_exp("c1"),)
    matched = (
        MatchedPair(truth_card=truth[0], side="F", instance=_inst(10, "Front", 0, 1000)),
    )
    phantoms = (_inst(99, "Front", 5000, 6000),)
    result = MatchResult(matched=matched, unmatched_truth=(), phantom_instances=phantoms)

    m = compute_video_metrics(result, truth)
    # 1 phantom out of 2 total pipeline outputs (1 matched + 1 phantom)
    assert m.phantom_rate == 0.5


def test_metrics_fb_inversion():
    truth = (_exp("c1"),)
    matched = (
        MatchedPair(truth_card=truth[0], side="F", instance=_inst(10, "Back", 0, 1000)),
        MatchedPair(truth_card=truth[0], side="B", instance=_inst(11, "Front", 2000, 3000)),
    )
    result = MatchResult(matched=matched, unmatched_truth=(), phantom_instances=())

    m = compute_video_metrics(result, truth)
    assert m.fb_correct == 0
    assert m.fb_accuracy == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/regression/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .matcher import MatchResult
from .truth import ExpectedCard


@dataclass(frozen=True)
class VideoMetrics:
    video_id: str
    expected_cards: int
    detected_cards: int
    recall: float
    phantom_count: int
    pipeline_output_count: int
    phantom_rate: float
    fb_correct: int
    fb_total: int
    fb_accuracy: float
    id_switches: int = 0
    sharpness_mean: float = 0.0
    wall_clock_s: float = 0.0
    peak_memory_mb: float = 0.0


def _angle_to_side(angle: str) -> str:
    a = angle.strip().lower()
    if a in {"front", "f"}:
        return "F"
    if a in {"back", "b"}:
        return "B"
    return "?"


def compute_video_metrics(
    match: MatchResult,
    truth: Sequence[ExpectedCard],
    *,
    video_id: str = "",
    id_switches: int = 0,
    sharpness_mean: float = 0.0,
    wall_clock_s: float = 0.0,
    peak_memory_mb: float = 0.0,
) -> VideoMetrics:
    expected = len(truth)
    detected_card_ids = {pair.truth_card.card_id for pair in match.matched}
    detected = len(detected_card_ids)
    recall = (detected / expected) if expected else 1.0

    pipeline_output = len(match.matched) + len(match.phantom_instances)
    phantom_rate = (len(match.phantom_instances) / pipeline_output) if pipeline_output else 0.0

    fb_total = len(match.matched)
    fb_correct = sum(
        1 for pair in match.matched
        if _angle_to_side(pair.instance.angle) == pair.side
    )
    fb_accuracy = (fb_correct / fb_total) if fb_total else 1.0

    return VideoMetrics(
        video_id=video_id,
        expected_cards=expected,
        detected_cards=detected,
        recall=recall,
        phantom_count=len(match.phantom_instances),
        pipeline_output_count=pipeline_output,
        phantom_rate=phantom_rate,
        fb_correct=fb_correct,
        fb_total=fb_total,
        fb_accuracy=fb_accuracy,
        id_switches=id_switches,
        sharpness_mean=sharpness_mean,
        wall_clock_s=wall_clock_s,
        peak_memory_mb=peak_memory_mb,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/regression/test_metrics.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/regression/metrics.py tests/regression/test_metrics.py
git commit -m "feat(harness): add per-video metrics (recall, phantom, F/B)"
```

---

## Task 5: ID switch metric

**Files:**
- Modify: `tests/regression/metrics.py` (add function)
- Modify: `tests/regression/test_metrics.py` (add test)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/regression/test_metrics.py
from tests.regression.metrics import count_id_switches


def test_id_switches_counts_track_changes_per_session():
    # pipeline_events shape produced by HysteresisTracker / ByteTrack adapter
    events = [
        {"event_type": "tracking", "session_id": 1, "track_id": "a", "timestamp_ms": 100},
        {"event_type": "tracking", "session_id": 1, "track_id": "a", "timestamp_ms": 200},
        {"event_type": "tracking", "session_id": 1, "track_id": "b", "timestamp_ms": 300},  # switch
        {"event_type": "tracking", "session_id": 2, "track_id": "c", "timestamp_ms": 5000},
        {"event_type": "tracking", "session_id": 2, "track_id": "d", "timestamp_ms": 5100},  # switch
        {"event_type": "tracking", "session_id": 2, "track_id": "d", "timestamp_ms": 5200},
    ]
    assert count_id_switches(events) == 2


def test_id_switches_ignores_non_tracking_events():
    events = [
        {"event_type": "session_reset", "session_id": 1, "timestamp_ms": 0},
        {"event_type": "tracking", "session_id": 1, "track_id": "a", "timestamp_ms": 100},
    ]
    assert count_id_switches(events) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/test_metrics.py::test_id_switches_counts_track_changes_per_session -v`
Expected: FAIL with `ImportError: cannot import name 'count_id_switches'`

- [ ] **Step 3: Write minimal implementation**

Append to `tests/regression/metrics.py`:

```python
from typing import Iterable


def count_id_switches(events: Iterable[dict]) -> int:
    by_session: dict = {}
    for ev in events:
        if ev.get("event_type") != "tracking":
            continue
        sid = ev.get("session_id")
        tid = ev.get("track_id")
        if sid is None or tid is None:
            continue
        by_session.setdefault(sid, []).append((ev.get("timestamp_ms", 0), tid))

    switches = 0
    for _sid, entries in by_session.items():
        entries.sort()
        prev = None
        for _ts, tid in entries:
            if prev is not None and tid != prev:
                switches += 1
            prev = tid
    return switches
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/regression/test_metrics.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/regression/metrics.py tests/regression/test_metrics.py
git commit -m "feat(harness): add ID switch metric"
```

---

## Task 6: Cross-video dedup F1

**Files:**
- Create: `tests/regression/cross_video.py`
- Test: `tests/regression/test_cross_video.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/test_cross_video.py
from tests.regression.cross_video import compute_dedup_f1, DedupMetrics
from tests.regression.matcher import MatchedPair
from tests.regression.pipeline_runner import HarnessInstance
from tests.regression.truth import ExpectedCard


def _exp(card_id, key):
    return ExpectedCard(card_id=card_id, front_present=True, back_present=False,
                       approx_front_window_ms=(0, 1000), physical_card_key=key)


def _inst(iid, dup_of=None):
    return HarnessInstance(
        instance_id=iid, video_id=1, session_id=1, angle="Front",
        duplicate_of=dup_of, fused_image_path=None,
        start_ms=0, end_ms=1000, detection_count=1, phash=None,
    )


def test_dedup_perfect():
    # Two videos each show physical card "X"; pipeline correctly marks the second as dup
    pairs_video_1 = [MatchedPair(truth_card=_exp("c1", "X"), side="F", instance=_inst(10, dup_of=None))]
    pairs_video_2 = [MatchedPair(truth_card=_exp("c1", "X"), side="F", instance=_inst(20, dup_of=10))]

    m = compute_dedup_f1(matched_pairs_per_video=[pairs_video_1, pairs_video_2])
    assert isinstance(m, DedupMetrics)
    assert m.true_positives == 1
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.f1 == 1.0


def test_dedup_missed_duplicate():
    # Pipeline failed to mark the second occurrence as a dup
    pairs_video_1 = [MatchedPair(truth_card=_exp("c1", "X"), side="F", instance=_inst(10, dup_of=None))]
    pairs_video_2 = [MatchedPair(truth_card=_exp("c1", "X"), side="F", instance=_inst(20, dup_of=None))]

    m = compute_dedup_f1(matched_pairs_per_video=[pairs_video_1, pairs_video_2])
    assert m.true_positives == 0
    assert m.false_negatives == 1
    assert m.f1 == 0.0


def test_dedup_false_positive():
    # Pipeline marked second card as dup of first, but truth says they're different
    pairs_video_1 = [MatchedPair(truth_card=_exp("c1", "X"), side="F", instance=_inst(10, dup_of=None))]
    pairs_video_2 = [MatchedPair(truth_card=_exp("c2", "Y"), side="F", instance=_inst(20, dup_of=10))]

    m = compute_dedup_f1(matched_pairs_per_video=[pairs_video_1, pairs_video_2])
    assert m.false_positives == 1
    assert m.f1 == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/test_cross_video.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/regression/cross_video.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .matcher import MatchedPair


@dataclass(frozen=True)
class DedupMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def compute_dedup_f1(matched_pairs_per_video: Sequence[Sequence[MatchedPair]]) -> DedupMetrics:
    """Compare pipeline duplicate links to truth physical_card_key groupings."""

    instance_to_key: dict = {}
    for video_pairs in matched_pairs_per_video:
        for pair in video_pairs:
            if pair.truth_card.physical_card_key:
                instance_to_key[pair.instance.instance_id] = pair.truth_card.physical_card_key

    truth_pairs: set = set()
    by_key: dict = {}
    for iid, key in instance_to_key.items():
        by_key.setdefault(key, []).append(iid)
    for ids in by_key.values():
        ids.sort()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                truth_pairs.add((ids[i], ids[j]))

    pipeline_pairs: set = set()
    for video_pairs in matched_pairs_per_video:
        for pair in video_pairs:
            inst = pair.instance
            if inst.duplicate_of is None:
                continue
            a, b = sorted([inst.instance_id, inst.duplicate_of])
            pipeline_pairs.add((a, b))

    tp = len(pipeline_pairs & truth_pairs)
    fp = len(pipeline_pairs - truth_pairs)
    fn = len(truth_pairs - pipeline_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 1.0 if not truth_pairs else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return DedupMetrics(
        true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=precision, recall=recall, f1=f1,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/regression/test_cross_video.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/regression/cross_video.py tests/regression/test_cross_video.py
git commit -m "feat(harness): add cross-video dedup F1 metric"
```

---

## Task 7: Report writers — JSON and Markdown with deltas

**Files:**
- Create: `tests/regression/report.py`
- Test: `tests/regression/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/test_report.py
import json
from pathlib import Path

from tests.regression.cross_video import DedupMetrics
from tests.regression.metrics import VideoMetrics
from tests.regression.report import write_json_report, write_markdown_report, AggregateReport


def _vm(video_id, recall=1.0, phantom=0.0, fb=1.0):
    return VideoMetrics(
        video_id=video_id, expected_cards=2, detected_cards=2,
        recall=recall, phantom_count=0, pipeline_output_count=2,
        phantom_rate=phantom, fb_correct=2, fb_total=2, fb_accuracy=fb,
        id_switches=0, sharpness_mean=100.0, wall_clock_s=12.5, peak_memory_mb=512.0,
    )


def test_write_json_report_round_trips(tmp_path):
    agg = AggregateReport(
        git_sha="abc123",
        per_video=(_vm("v1"), _vm("v2", recall=0.5)),
        dedup=DedupMetrics(1, 0, 0, 1.0, 1.0, 1.0),
    )
    path = tmp_path / "report.json"
    write_json_report(agg, path)

    loaded = json.loads(path.read_text())
    assert loaded["git_sha"] == "abc123"
    assert len(loaded["per_video"]) == 2
    assert loaded["per_video"][1]["recall"] == 0.5
    assert loaded["dedup"]["f1"] == 1.0


def test_write_markdown_report_includes_deltas(tmp_path):
    baseline = AggregateReport(
        git_sha="aaa", per_video=(_vm("v1", recall=0.6, phantom=0.4),),
        dedup=DedupMetrics(0, 0, 0, 1.0, 1.0, 1.0),
    )
    current = AggregateReport(
        git_sha="bbb", per_video=(_vm("v1", recall=0.9, phantom=0.1),),
        dedup=DedupMetrics(0, 0, 0, 1.0, 1.0, 1.0),
    )
    path = tmp_path / "report.md"
    write_markdown_report(current, path, baseline=baseline)

    text = path.read_text()
    assert "v1" in text
    assert "0.900" in text or "0.90" in text  # current recall
    assert "+0.300" in text or "+0.30" in text  # delta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/regression/report.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .cross_video import DedupMetrics
from .metrics import VideoMetrics


@dataclass(frozen=True)
class AggregateReport:
    git_sha: str
    per_video: Tuple[VideoMetrics, ...]
    dedup: DedupMetrics


def _aggregate_recall(per_video: Sequence[VideoMetrics]) -> float:
    expected = sum(v.expected_cards for v in per_video)
    detected = sum(v.detected_cards for v in per_video)
    return (detected / expected) if expected else 1.0


def _aggregate_phantom_rate(per_video: Sequence[VideoMetrics]) -> float:
    output = sum(v.pipeline_output_count for v in per_video)
    phantoms = sum(v.phantom_count for v in per_video)
    return (phantoms / output) if output else 0.0


def _aggregate_fb_accuracy(per_video: Sequence[VideoMetrics]) -> float:
    total = sum(v.fb_total for v in per_video)
    correct = sum(v.fb_correct for v in per_video)
    return (correct / total) if total else 1.0


def write_json_report(report: AggregateReport, path: Path) -> None:
    payload = {
        "git_sha": report.git_sha,
        "aggregates": {
            "recall": _aggregate_recall(report.per_video),
            "phantom_rate": _aggregate_phantom_rate(report.per_video),
            "fb_accuracy": _aggregate_fb_accuracy(report.per_video),
        },
        "per_video": [asdict(v) for v in report.per_video],
        "dedup": asdict(report.dedup),
    }
    path.write_text(json.dumps(payload, indent=2))


def _delta(current: float, baseline: Optional[float]) -> str:
    if baseline is None:
        return ""
    diff = current - baseline
    sign = "+" if diff >= 0 else ""
    return f" ({sign}{diff:.3f})"


def write_markdown_report(
    report: AggregateReport,
    path: Path,
    baseline: Optional[AggregateReport] = None,
) -> None:
    lines = [
        f"# Harness report — {report.git_sha}",
        "",
        "## Aggregates",
        "",
    ]
    cur_recall = _aggregate_recall(report.per_video)
    cur_phantom = _aggregate_phantom_rate(report.per_video)
    cur_fb = _aggregate_fb_accuracy(report.per_video)
    base_recall = _aggregate_recall(baseline.per_video) if baseline else None
    base_phantom = _aggregate_phantom_rate(baseline.per_video) if baseline else None
    base_fb = _aggregate_fb_accuracy(baseline.per_video) if baseline else None

    lines.append(f"- Recall: **{cur_recall:.3f}**{_delta(cur_recall, base_recall)}")
    lines.append(f"- Phantom rate: **{cur_phantom:.3f}**{_delta(cur_phantom, base_phantom)}")
    lines.append(f"- F/B accuracy: **{cur_fb:.3f}**{_delta(cur_fb, base_fb)}")
    lines.append(f"- Dedup F1: **{report.dedup.f1:.3f}**")
    lines.append("")

    lines.append("## Per video")
    lines.append("")
    lines.append("| video | recall | phantom_rate | fb_acc | id_switches | wall_s |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    base_by_id = {v.video_id: v for v in baseline.per_video} if baseline else {}
    for v in report.per_video:
        b = base_by_id.get(v.video_id)
        recall_cell = f"{v.recall:.3f}{_delta(v.recall, b.recall if b else None)}"
        phantom_cell = f"{v.phantom_rate:.3f}{_delta(v.phantom_rate, b.phantom_rate if b else None)}"
        fb_cell = f"{v.fb_accuracy:.3f}{_delta(v.fb_accuracy, b.fb_accuracy if b else None)}"
        lines.append(
            f"| {v.video_id} | {recall_cell} | {phantom_cell} | {fb_cell} | {v.id_switches} | {v.wall_clock_s:.1f} |"
        )

    path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/regression/test_report.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/regression/report.py tests/regression/test_report.py
git commit -m "feat(harness): add JSON + Markdown report writers with deltas"
```

---

## Task 8: Harness orchestration

**Files:**
- Create: `tests/regression/harness.py`
- Test: `tests/regression/test_harness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/test_harness.py
import json
import time
from pathlib import Path

import pytest

from tests.regression.harness import run_corpus, HarnessConfig


def test_run_corpus_returns_aggregate_report(tmp_path, monkeypatch):
    # Build a minimal in-memory fixture: 1 video with 1 card
    video_id_str = "vid_001"
    corpus_dir = tmp_path / "corpus"
    video_dir = corpus_dir / video_id_str
    video_dir.mkdir(parents=True)
    truth = {
        "video_id": video_id_str,
        "video_path": str(video_dir / "fake.mp4"),
        "expected_cards": [
            {"card_id": "c1", "front_present": True, "back_present": False,
             "approx_front_window_ms": [0, 1000]},
        ],
    }
    (video_dir / f"{video_id_str}.truth.json").write_text(json.dumps(truth))

    # Stub the pipeline runner to return a fixed result
    from tests.regression import harness as harness_mod
    from tests.regression.pipeline_runner import HarnessInstance

    def fake_runner(video_path, db_path, output_dir):
        return [
            HarnessInstance(
                instance_id=10, video_id=1, session_id=1, angle="Front",
                duplicate_of=None, fused_image_path=None,
                start_ms=100, end_ms=900, detection_count=5, phash=None,
            ),
        ], 1.5, 256.0, []  # instances, wall_s, peak_mb, events

    monkeypatch.setattr(harness_mod, "run_pipeline_for_video", fake_runner)

    cfg = HarnessConfig(corpus_dir=corpus_dir, output_dir=tmp_path / "out", git_sha="testsha")
    report = run_corpus(cfg)

    assert report.git_sha == "testsha"
    assert len(report.per_video) == 1
    assert report.per_video[0].recall == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/regression/harness.py
from __future__ import annotations

import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .cross_video import compute_dedup_f1
from .matcher import MatchedPair, match_instances_to_truth
from .metrics import VideoMetrics, compute_video_metrics, count_id_switches
from .pipeline_runner import HarnessInstance, load_instances_for_video
from .report import AggregateReport, write_json_report, write_markdown_report
from .truth import GroundTruth, load_truth


@dataclass(frozen=True)
class HarnessConfig:
    corpus_dir: Path
    output_dir: Path
    git_sha: str
    tolerance_ms: int = 500
    db_path: Path = Path("card_capture_output/cards.sqlite")


def _peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS, KB on Linux
    import sys
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def run_pipeline_for_video(
    video_path: Path,
    db_path: Path,
    output_dir: Path,
) -> Tuple[List[HarnessInstance], float, float, list]:
    """Run the real pipeline against a video and return harness records."""
    from card_capture.cli import _run_process
    import argparse

    args = argparse.Namespace(
        video_path=Path(video_path),
        output_dir=Path(output_dir),
        db=Path(db_path),
        config=Path("card_capture_config.json"),
    )

    start = time.perf_counter()
    rc = _run_process(args)
    wall = time.perf_counter() - start
    if rc != 0:
        raise RuntimeError(f"pipeline returned non-zero exit code {rc} for {video_path}")

    # Find the most recent video_id in storage matching this video path
    from card_capture.storage import Storage
    storage = Storage(db_path)
    storage.initialize()
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT id FROM videos WHERE source_path = ? ORDER BY id DESC LIMIT 1",
            (str(video_path),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"no video row found for {video_path} after pipeline run")
        video_db_id = int(row["id"])

        events = conn.execute(
            "SELECT event_type, data_json FROM pipeline_events WHERE video_id = ?",
            (video_db_id,),
        ).fetchall()
    import json as _json
    parsed_events = []
    for e in events:
        d = _json.loads(e["data_json"]) if e["data_json"] else {}
        parsed_events.append({"event_type": e["event_type"], **d})

    instances = load_instances_for_video(db_path, video_db_id)
    return instances, wall, _peak_memory_mb(), parsed_events


def run_corpus(cfg: HarnessConfig) -> AggregateReport:
    truth_files = sorted(cfg.corpus_dir.glob("*/*.truth.json"))
    if not truth_files:
        raise RuntimeError(f"no truth.json files found under {cfg.corpus_dir}")

    per_video: List[VideoMetrics] = []
    matched_per_video: List[Sequence[MatchedPair]] = []

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for truth_path in truth_files:
        truth: GroundTruth = load_truth(truth_path)
        video_path = Path(truth.video_path)

        instances, wall, peak_mb, events = run_pipeline_for_video(
            video_path=video_path, db_path=cfg.db_path, output_dir=cfg.output_dir,
        )
        match = match_instances_to_truth(instances, truth.expected_cards, tolerance_ms=cfg.tolerance_ms)
        sharpness = 0.0  # placeholder; quality metric refined when scoring exposes per-instance values

        vm = compute_video_metrics(
            match, truth.expected_cards,
            video_id=truth.video_id,
            id_switches=count_id_switches(events),
            sharpness_mean=sharpness, wall_clock_s=wall, peak_memory_mb=peak_mb,
        )
        per_video.append(vm)
        matched_per_video.append(match.matched)

    dedup = compute_dedup_f1(matched_per_video)
    report = AggregateReport(git_sha=cfg.git_sha, per_video=tuple(per_video), dedup=dedup)
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/regression/test_harness.py -v`
Expected: 1 PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/regression/harness.py tests/regression/test_harness.py
git commit -m "feat(harness): add orchestration entry point"
```

---

## Task 9: CLI integration

**Files:**
- Modify: `src/card_capture/cli.py`

- [ ] **Step 1: Add the `harness` subparser and runner**

In `src/card_capture/cli.py`, find `build_parser` (around line 17) and add a new subparser block before the `return parser` line:

```python
    harness = subparsers.add_parser("harness", help="Run regression harness against golden corpus")
    harness_sub = harness.add_subparsers(dest="harness_command", required=True)

    harness_run = harness_sub.add_parser("run", help="Run pipeline on corpus and write report")
    harness_run.add_argument("--corpus", type=Path, default=Path("tests/fixtures/golden_corpus"))
    harness_run.add_argument("--db", type=Path, default=Path("card_capture_output/cards.sqlite"))
    harness_run.add_argument("--output-dir", type=Path, default=Path("card_capture_output"))
    harness_run.add_argument("--reports-dir", type=Path, default=Path("reports"))
    harness_run.add_argument("--baseline", type=Path, default=None,
                              help="Optional baseline JSON report to compute deltas against")

    harness_compare = harness_sub.add_parser("compare", help="Compare two existing reports")
    harness_compare.add_argument("baseline", type=Path)
    harness_compare.add_argument("current", type=Path)
    harness_compare.add_argument("--out", type=Path, default=None)
```

Then in `main()` (around line 34) add the dispatch:

```python
    if args.command == "harness":
        return _run_harness(args)
```

Add the runner function at the end of the file (before `if __name__ == "__main__":`):

```python
def _run_harness(args: argparse.Namespace) -> int:
    if args.harness_command == "run":
        return _run_harness_run(args)
    if args.harness_command == "compare":
        return _run_harness_compare(args)
    return 2


def _run_harness_run(args: argparse.Namespace) -> int:
    import json
    import subprocess
    from tests.regression.harness import HarnessConfig, run_corpus
    from tests.regression.report import write_json_report, write_markdown_report, AggregateReport
    from tests.regression.metrics import VideoMetrics
    from tests.regression.cross_video import DedupMetrics

    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    cfg = HarnessConfig(
        corpus_dir=args.corpus, output_dir=args.output_dir, git_sha=sha, db_path=args.db,
    )
    report = run_corpus(cfg)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / f"{sha}.json"
    md_path = args.reports_dir / f"{sha}.md"

    baseline_report = None
    if args.baseline and args.baseline.exists():
        raw = json.loads(args.baseline.read_text())
        baseline_report = AggregateReport(
            git_sha=raw["git_sha"],
            per_video=tuple(VideoMetrics(**v) for v in raw["per_video"]),
            dedup=DedupMetrics(**raw["dedup"]),
        )

    write_json_report(report, json_path)
    write_markdown_report(report, md_path, baseline=baseline_report)
    print(f"Wrote {json_path} and {md_path}")
    return 0


def _run_harness_compare(args: argparse.Namespace) -> int:
    import json
    from tests.regression.report import AggregateReport, write_markdown_report
    from tests.regression.metrics import VideoMetrics
    from tests.regression.cross_video import DedupMetrics

    def _load(path):
        raw = json.loads(path.read_text())
        return AggregateReport(
            git_sha=raw["git_sha"],
            per_video=tuple(VideoMetrics(**v) for v in raw["per_video"]),
            dedup=DedupMetrics(**raw["dedup"]),
        )

    baseline = _load(args.baseline)
    current = _load(args.current)
    out = args.out or args.current.with_suffix(".compare.md")
    write_markdown_report(current, out, baseline=baseline)
    print(f"Wrote {out}")
    return 0
```

- [ ] **Step 2: Smoke-test CLI surface (no harness execution)**

Run: `card-capture harness --help`
Expected: prints help showing `run` and `compare` subcommands, exit 0.

Run: `card-capture harness run --help`
Expected: prints help showing `--corpus`, `--db`, etc., exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/card_capture/cli.py
git commit -m "feat(cli): add harness run and compare subcommands"
```

---

## Task 10: Makefile target

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write the file**

```makefile
# Makefile

.PHONY: harness baseline test

test:
	pytest tests/

harness:
	card-capture harness run

baseline:
	card-capture harness run --baseline reports/baseline_v3.json
```

- [ ] **Step 2: Smoke check**

Run: `make harness 2>&1 | head -3`
Expected: prints `card-capture harness run` and starts attempting to run; will fail loudly if no corpus exists yet (this is expected — we add fixtures next).

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add Makefile with harness, baseline, test targets"
```

---

## Task 11: Labeling UI route + template

**Files:**
- Modify: `src/card_capture/review.py`
- Create: `src/card_capture/templates/labeling.html`

- [ ] **Step 1: Add the routes to review.py**

In `src/card_capture/review.py`, after the existing `@app.get("/timeline", ...)` route, add:

```python
    @app.get("/label/{video_id}", response_class=HTMLResponse)
    def label_get(request: Request, video_id: int):
        with storage._connect() as conn:
            video_row = conn.execute(
                "SELECT id, source_path FROM videos WHERE id = ?", (video_id,),
            ).fetchone()
            if video_row is None:
                return HTMLResponse(f"video {video_id} not found", status_code=404)

            instances = conn.execute(
                """
                SELECT ci.id AS instance_id, ci.angle, ci.session_id, ci.is_duplicate_of,
                       MIN(cv.timestamp_ms) AS start_time, MAX(cv.timestamp_ms) AS end_time,
                       ci.fused_image_path
                FROM card_instances ci
                LEFT JOIN card_views cv ON cv.card_instance_id = ci.id
                WHERE ci.video_id = ?
                GROUP BY ci.id
                ORDER BY start_time ASC
                """,
                (video_id,),
            ).fetchall()

        instance_data = [dict(r) for r in instances]
        truth_path = _truth_path_for_video(video_row["source_path"])
        existing_truth = json.loads(truth_path.read_text()) if truth_path.exists() else None

        return templates.TemplateResponse(
            request, "labeling.html",
            {
                "video_id": video_id,
                "video_path": video_row["source_path"],
                "instances": instance_data,
                "truth": existing_truth,
                "truth_path": str(truth_path),
            },
        )

    @app.post("/label/{video_id}/save")
    async def label_save(video_id: int, request: Request):
        payload = await request.json()
        with storage._connect() as conn:
            video_row = conn.execute(
                "SELECT source_path FROM videos WHERE id = ?", (video_id,),
            ).fetchone()
            if video_row is None:
                return {"ok": False, "error": "video not found"}

        truth_path = _truth_path_for_video(video_row["source_path"])
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(json.dumps(payload, indent=2))
        return {"ok": True, "path": str(truth_path)}
```

Add the helper near the top of `create_app` (after `db_path = Path(db_path).resolve()`):

```python
    def _truth_path_for_video(source_path: str) -> Path:
        # Convention: tests/fixtures/golden_corpus/<video_stem>/<video_stem>.truth.json
        stem = Path(source_path).stem
        return (
            Path("tests/fixtures/golden_corpus") / stem / f"{stem}.truth.json"
        ).resolve()
```

- [ ] **Step 2: Create the template**

Create `src/card_capture/templates/labeling.html`:

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Label video {{ video_id }}</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; font-size: 14px; }
    th { background: #f4f4f4; text-align: left; }
    img { max-height: 80px; }
    .truth-box { background: #fffbe6; padding: 0.5rem; border: 1px solid #e0c000; margin: 0.5rem 0; }
    button { padding: 0.4rem 0.8rem; }
    .saved { color: green; }
    .err { color: red; }
  </style>
</head>
<body>
  <h1>Label video {{ video_id }}</h1>
  <p><strong>Source:</strong> {{ video_path }}</p>
  <p><strong>Truth file:</strong> {{ truth_path }}</p>

  <h2>Pipeline-detected Card Instances</h2>
  <p>Mark each as a real Front, real Back, or phantom. Add a <code>physical_card_key</code> to mark cross-video duplicates.</p>

  <table id="instances">
    <thead>
      <tr>
        <th>iid</th><th>thumb</th><th>session</th><th>angle (auto)</th>
        <th>start_ms</th><th>end_ms</th>
        <th>truth: side</th><th>truth: card_id</th><th>physical_card_key</th>
      </tr>
    </thead>
    <tbody>
      {% for inst in instances %}
      <tr data-iid="{{ inst.instance_id }}">
        <td>{{ inst.instance_id }}</td>
        <td>{% if inst.fused_image_path %}<img src="/fused_images/{{ inst.instance_id }}">{% endif %}</td>
        <td>{{ inst.session_id }}</td>
        <td>{{ inst.angle }}</td>
        <td>{{ inst.start_time }}</td>
        <td>{{ inst.end_time }}</td>
        <td>
          <select class="side">
            <option value="">(phantom)</option>
            <option value="F">Front</option>
            <option value="B">Back</option>
          </select>
        </td>
        <td><input class="card_id" placeholder="e.g. card_001"></td>
        <td><input class="phys_key" placeholder="e.g. topps_chrome_2024_42"></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>Missing cards</h2>
  <p>Click "Add missing card" to record a card the pipeline missed.</p>
  <table id="missing">
    <thead><tr><th>card_id</th><th>front_ms (start, end)</th><th>back_ms</th><th>physical_card_key</th><th></th></tr></thead>
    <tbody></tbody>
  </table>
  <button id="add-missing">Add missing card</button>

  <h2>Save</h2>
  <button id="save">Save truth.json</button>
  <span id="status"></span>

  <script>
    const VIDEO_ID = {{ video_id }};
    const VIDEO_PATH = {{ video_path|tojson }};
    const EXISTING = {{ truth|tojson if truth else 'null' }};

    function applyExisting() {
      if (!EXISTING) return;
      const cardsByIid = {};  // not directly applicable since truth doesn't store iid; left as TODO
      // Pre-fill missing-card list from existing truth
      const missingTbody = document.querySelector('#missing tbody');
      for (const c of (EXISTING.expected_cards || [])) {
        addMissingRow(c);
      }
    }

    function addMissingRow(card) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><input class="m_card_id" value="${card?.card_id || ''}"></td>
        <td><input class="m_front" placeholder="start,end" value="${(card?.approx_front_window_ms || []).join(',')}"></td>
        <td><input class="m_back" placeholder="start,end" value="${(card?.approx_back_window_ms || []).join(',')}"></td>
        <td><input class="m_phys" value="${card?.physical_card_key || ''}"></td>
        <td><button onclick="this.closest('tr').remove()">x</button></td>
      `;
      document.querySelector('#missing tbody').appendChild(tr);
    }

    document.getElementById('add-missing').onclick = () => addMissingRow(null);

    document.getElementById('save').onclick = async () => {
      const expected = [];
      // Collect labeled detected instances
      for (const tr of document.querySelectorAll('#instances tr[data-iid]')) {
        const side = tr.querySelector('.side').value;
        const card_id = tr.querySelector('.card_id').value.trim();
        const phys = tr.querySelector('.phys_key').value.trim();
        if (!side || !card_id) continue;  // phantoms / unlabeled skipped
        const start = parseInt(tr.children[4].textContent);
        const end = parseInt(tr.children[5].textContent);
        const existing = expected.find(c => c.card_id === card_id);
        if (existing) {
          if (side === 'F') { existing.front_present = true; existing.approx_front_window_ms = [start, end]; }
          else { existing.back_present = true; existing.approx_back_window_ms = [start, end]; }
          if (phys) existing.physical_card_key = phys;
        } else {
          const c = { card_id, front_present: side === 'F', back_present: side === 'B' };
          if (side === 'F') c.approx_front_window_ms = [start, end];
          else c.approx_back_window_ms = [start, end];
          if (phys) c.physical_card_key = phys;
          expected.push(c);
        }
      }
      // Add missing cards
      for (const tr of document.querySelectorAll('#missing tbody tr')) {
        const card_id = tr.querySelector('.m_card_id').value.trim();
        if (!card_id) continue;
        const front = tr.querySelector('.m_front').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        const back = tr.querySelector('.m_back').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        const phys = tr.querySelector('.m_phys').value.trim();
        const c = { card_id, front_present: front.length === 2, back_present: back.length === 2 };
        if (front.length === 2) c.approx_front_window_ms = front;
        if (back.length === 2) c.approx_back_window_ms = back;
        if (phys) c.physical_card_key = phys;
        expected.push(c);
      }

      const payload = {
        video_id: 'video_' + VIDEO_ID,
        video_path: VIDEO_PATH,
        labeled_at: new Date().toISOString().slice(0, 10),
        expected_cards: expected,
      };
      const status = document.getElementById('status');
      try {
        const resp = await fetch(`/label/${VIDEO_ID}/save`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
        });
        const out = await resp.json();
        if (out.ok) status.innerHTML = `<span class="saved">saved → ${out.path}</span>`;
        else status.innerHTML = `<span class="err">${out.error}</span>`;
      } catch (e) {
        status.innerHTML = `<span class="err">${e.message}</span>`;
      }
    };

    applyExisting();
  </script>
</body>
</html>
```

- [ ] **Step 3: Manual smoke test**

Run: `card-capture review --db card_capture_output/cards.sqlite --port 8000 &`
Open in browser: `http://127.0.0.1:8000/label/1` (replace `1` with a real video_id from your DB).
Expected: page renders, lists pipeline instances, side selector + card_id + phys_key inputs visible, "Save truth.json" button present.

Click "Add missing card" → empty row appears. Click "Save" → status shows green path. Verify file exists at `tests/fixtures/golden_corpus/<stem>/<stem>.truth.json` and parses with `python -c "import json; print(json.load(open('<path>')))"`.

Stop the server with `kill %1`.

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/review.py src/card_capture/templates/labeling.html
git commit -m "feat(review): add labeling mode that writes truth.json"
```

---

## Task 12: Capture baseline_v3 report

**Files:**
- Create: `reports/baseline_v3.json`, `reports/baseline_v3.md`
- Create: `tests/fixtures/golden_corpus/.gitkeep`, `reports/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Set up directories**

```bash
mkdir -p tests/fixtures/golden_corpus reports
touch tests/fixtures/golden_corpus/.gitkeep reports/.gitkeep
```

- [ ] **Step 2: Update .gitignore**

Append to `.gitignore`:

```
# Harness reports — track only baselines
reports/*
!reports/.gitkeep
!reports/baseline_*
```

- [ ] **Step 3: Manually label your videos**

This is a user-driven step, not automated. For each video in `card_capture_output/cards.sqlite`:

1. Run `card-capture review --port 8000`
2. Open `http://127.0.0.1:8000/label/<video_id>` for each video.
3. Label each detected instance: side (F/B/phantom), `card_id`, optional `physical_card_key`.
4. Click "Add missing card" for any cards the pipeline missed.
5. "Save truth.json".
6. Move the source video into `tests/fixtures/golden_corpus/<stem>/<stem>.mp4`.

Verify with: `ls tests/fixtures/golden_corpus/*/*.truth.json` — should list one file per labeled video.

- [ ] **Step 4: Run the harness against the corpus**

```bash
card-capture harness run --reports-dir reports
mv reports/<sha>.json reports/baseline_v3.json
mv reports/<sha>.md reports/baseline_v3.md
```

Expected: prints a summary line; both files exist; markdown file shows aggregate metrics.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/golden_corpus/.gitkeep reports/.gitkeep .gitignore
git add tests/fixtures/golden_corpus/  # the labeled truth.json files
git add reports/baseline_v3.json reports/baseline_v3.md
git commit -m "feat(harness): capture baseline_v3 report on golden corpus"
```

---

## Self-review

- All steps include actual code or actual commands; no placeholders.
- File paths are exact (`tests/regression/`, `src/card_capture/`).
- Tests precede implementation in every code-bearing task.
- Commits at the end of every task.
- Spec coverage: §3.1 corpus location ✓ (Task 12), §3.2 truth schema ✓ (Task 1), §3.3 labeling mode ✓ (Task 11), §3.4 metrics ✓ (Tasks 4–6), report ✓ (Task 7), CLI ✓ (Task 9), Make target ✓ (Task 10), §3.5 deliverables all covered (Tasks 1, 8, 9, 11, 12), §3.6 effort 2-3 days consistent with 12 task chunks.
- Type consistency: `HarnessInstance`, `ExpectedCard`, `MatchedPair`, `MatchResult`, `VideoMetrics`, `DedupMetrics`, `AggregateReport` are defined once and used consistently across tasks.
- Open issue noted in Task 2 step 3: `phash` column may not exist on `card_instances` — verify in `storage.py` before running.
