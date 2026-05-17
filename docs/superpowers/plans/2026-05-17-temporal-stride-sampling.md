# Temporal Stride Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace top-N-by-presence-score frame selection with duration-proportional temporal bucketing to reduce YOLO frames by 3-4× with no quality regression.

**Architecture:** Add `target_yolo_fps` config field (default 3.0) that controls how many frames per second of card presence go to YOLO. `_score_sharpness_in_window` divides each presence window into `ceil(duration_s * target_yolo_fps)` equal temporal buckets and picks the highest-presence-score frame from each bucket. `max_candidates_per_window` default drops from 48 → 24 as a safety cap.

**Tech Stack:** Python, existing `AdaptivePresenceSampler` in `src/card_capture/sampler/__init__.py`, Metaflow pipeline config wiring.

**Spec:** `docs/superpowers/specs/2026-05-17-temporal-stride-sampling-design.md`

---

### Task 1: Add `target_yolo_fps` to config and RunContext

**Files:**
- Modify: `src/card_capture/config.py:34`
- Modify: `pipeline/steps/start.py:50` (RunContext field) and `pipeline/steps/start.py:151` (init_run wiring)

- [ ] **Step 1: Add `target_yolo_fps` to `PipelineConfig`**

In `src/card_capture/config.py`, after line 34 (`confirm_scan_fps: float = 5.0`):

```python
    confirm_scan_fps: float = 5.0
    target_yolo_fps: float = 3.0
```

- [ ] **Step 2: Add `target_yolo_fps` to `RunContext`**

In `pipeline/steps/start.py`, after line 50 (`confirm_scan_fps: float = 5.0`):

```python
    confirm_scan_fps: float = 5.0
    target_yolo_fps: float = 3.0
```

- [ ] **Step 3: Wire `target_yolo_fps` into `init_run`**

In `pipeline/steps/start.py`, after line 151 (`confirm_scan_fps=cfg.confirm_scan_fps,`):

```python
        confirm_scan_fps=cfg.confirm_scan_fps,
        target_yolo_fps=cfg.target_yolo_fps,
```

- [ ] **Step 4: Run existing tests to confirm no breakage**

```bash
python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py 2>&1 | tail -20
```

Expected: same pass/fail counts as before this task (pre-existing failures are documented in CLAUDE.md).

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/config.py pipeline/steps/start.py
git commit -m "feat(config): add target_yolo_fps field (default 3.0)"
```

---

### Task 2: Write failing test for temporal stride selection

**Files:**
- Test: `tests/test_sampler_fast_scan.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sampler_fast_scan.py`:

```python
def _make_presence_window(start_frame, end_frame, n_records, base_score=0.8):
    """Helper: build a PresenceWindow and matching _AdaptiveScanFrame list."""
    from card_capture.sampler import PresenceWindow, _AdaptiveScanFrame
    window = PresenceWindow(start_frame=start_frame, end_frame=end_frame)
    step = (end_frame - start_frame) // max(n_records - 1, 1)
    records = []
    for i in range(n_records):
        fi = start_frame + i * step
        records.append(_AdaptiveScanFrame(
            frame_index=fi,
            timestamp_ms=fi * 16,
            image=np.zeros((12, 12, 3), dtype=np.uint8),
            metrics={"edge_density": 1.0},
            presence_score=base_score + (i % 3) * 0.05,
        ))
    return window, records


def test_score_sharpness_temporal_spread():
    """Candidates must be spread across the window, not clustered at high-score frames."""
    sampler = AdaptivePresenceSampler(target_yolo_fps=3.0)
    sampler.last_source_fps = 60.0
    # 10-second window at 60fps = frames 0..600; 60 scan records
    window, records = _make_presence_window(0, 600, n_records=60)
    sampler._scan_frames = records

    result = sampler._score_sharpness_in_window(window)
    candidates = [fi for fi, _ in result.frame_candidates]

    # At 3fps over 10s → ~30 candidates, capped at max_candidates_per_window=24
    assert 10 <= len(candidates) <= 24

    # Candidates must span at least 80% of the window range
    assert max(candidates) - min(candidates) >= 0.8 * 600

    # No two consecutive candidates come from the same 5% of the window
    segment = 600 / 20
    segments_used = {int(fi / segment) for fi in candidates}
    assert len(segments_used) >= 10, "candidates are clustered, not temporally spread"


def test_score_sharpness_short_window_respects_min():
    """A very short window still yields at least min_candidates_per_window frames."""
    sampler = AdaptivePresenceSampler(target_yolo_fps=3.0)
    sampler.last_source_fps = 60.0
    # 0.5-second window → ceil(0.5 * 3) = 2, but min is 3
    window, records = _make_presence_window(0, 30, n_records=5)
    sampler._scan_frames = records

    result = sampler._score_sharpness_in_window(window)
    assert len(result.frame_candidates) >= sampler.min_candidates_per_window


def test_score_sharpness_picks_best_score_per_bucket():
    """Within each bucket, the frame with the highest presence_score is chosen."""
    from card_capture.sampler import PresenceWindow, _AdaptiveScanFrame
    sampler = AdaptivePresenceSampler(target_yolo_fps=1.0, max_candidates_per_window=2)
    sampler.last_source_fps = 60.0
    # 2-second window → 2 buckets; plant known best scores
    window = PresenceWindow(start_frame=0, end_frame=120)
    records = [
        _AdaptiveScanFrame(0,  0,  np.zeros((4,4,3), dtype=np.uint8), {}, presence_score=0.5),
        _AdaptiveScanFrame(30, 500, np.zeros((4,4,3), dtype=np.uint8), {}, presence_score=0.9),  # best in bucket 0
        _AdaptiveScanFrame(60, 1000, np.zeros((4,4,3), dtype=np.uint8), {}, presence_score=0.6),
        _AdaptiveScanFrame(90, 1500, np.zeros((4,4,3), dtype=np.uint8), {}, presence_score=0.95), # best in bucket 1
        _AdaptiveScanFrame(120,2000, np.zeros((4,4,3), dtype=np.uint8), {}, presence_score=0.4),
    ]
    sampler._scan_frames = records

    result = sampler._score_sharpness_in_window(window)
    chosen_indices = [fi for fi, _ in result.frame_candidates]
    assert 30 in chosen_indices, "highest-score frame in bucket 0 must be chosen"
    assert 90 in chosen_indices, "highest-score frame in bucket 1 must be chosen"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py::test_score_sharpness_temporal_spread tests/test_sampler_fast_scan.py::test_score_sharpness_short_window_respects_min tests/test_sampler_fast_scan.py::test_score_sharpness_picks_best_score_per_bucket -v 2>&1 | tail -20
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'target_yolo_fps'`

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_sampler_fast_scan.py
git commit -m "test(sampler): failing tests for temporal stride selection"
```

---

### Task 3: Implement temporal stride in `AdaptivePresenceSampler`

**Files:**
- Modify: `src/card_capture/sampler/__init__.py:380` (`__init__` signature)
- Modify: `src/card_capture/sampler/__init__.py:407` (self assignments)
- Modify: `src/card_capture/sampler/__init__.py:706-738` (`_score_sharpness_in_window`)

- [ ] **Step 1: Add `target_yolo_fps` param and `last_source_fps` default to `__init__`**

In `src/card_capture/sampler/__init__.py`, the `AdaptivePresenceSampler.__init__` signature starts around line 374. Change the `max_candidates_per_window` default and add `target_yolo_fps`:

```python
        max_candidates_per_window: int = 24,
```

(was `48` — change the default only, same line)

After `confirm_scan_fps: Optional[float] = None,` add:

```python
        target_yolo_fps: float = 3.0,
```

In the body of `__init__`, after the existing `self.max_candidates_per_window = max(...)` lines, add:

```python
        self.target_yolo_fps = max(0.1, target_yolo_fps)
        self.last_source_fps: float = 30.0  # overwritten by _scan_video; safe default for tests
```

- [ ] **Step 2: Replace `_score_sharpness_in_window`**

Replace the entire method (lines 706-738) with:

```python
    def _score_sharpness_in_window(self, window: PresenceWindow) -> PresenceWindow:
        records = self._candidate_records_for_window(window)
        if not records:
            return window

        source_fps = getattr(self, 'last_source_fps', 30.0) or 30.0
        duration_s = (window.end_frame - window.start_frame) / source_fps
        target = math.ceil(duration_s * self.target_yolo_fps)
        target = max(self.min_candidates_per_window, min(self.max_candidates_per_window, target))
        target = min(target, len(records))

        sorted_records = sorted(records, key=lambda r: r.frame_index)

        if len(sorted_records) <= target:
            window.frame_candidates = [
                (r.frame_index, r.presence_score) for r in sorted_records
            ]
            return window

        # Divide into target equal temporal buckets; pick highest presence_score per bucket
        n = len(sorted_records)
        selected = []
        for i in range(target):
            start = int(i * n / target)
            end = int((i + 1) * n / target)
            bucket = sorted_records[start:end]
            if bucket:
                best = max(bucket, key=lambda r: r.presence_score)
                selected.append((best.frame_index, best.presence_score))

        window.frame_candidates = selected
        return window
```

- [ ] **Step 3: Run the three new tests**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py::test_score_sharpness_temporal_spread tests/test_sampler_fast_scan.py::test_score_sharpness_short_window_respects_min tests/test_sampler_fast_scan.py::test_score_sharpness_picks_best_score_per_bucket -v 2>&1 | tail -20
```

Expected: all three PASS.

- [ ] **Step 4: Run the full fast-scan test file**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py -v 2>&1 | tail -20
```

Expected: all existing tests still pass.

- [ ] **Step 5: Run the broader test suite**

```bash
python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py 2>&1 | tail -20
```

Expected: same pass/fail counts as before Task 1 (pre-existing failures noted in CLAUDE.md are not regressions).

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/sampler/__init__.py
git commit -m "feat(sampler): duration-proportional temporal stride selection

Replaces top-N-by-presence-score with temporal bucketing:
target = ceil(window_seconds * target_yolo_fps), capped at
max_candidates_per_window (now 24). Each bucket picks the
highest-presence-score frame. Estimated 3-4x YOLO frame reduction."
```

---

### Task 4: Wire `target_yolo_fps` through detect step + telemetry

**Files:**
- Modify: `pipeline/steps/detect.py:165-171` (`_build_sampler_detector`)
- Modify: `src/card_capture/workers.py:478-487` (telemetry attrs tuple)

- [ ] **Step 1: Pass `target_yolo_fps` to `AdaptivePresenceSampler` in `detect.py`**

Replace the `AdaptivePresenceSampler(...)` constructor call in `_build_sampler_detector` (lines 165-171):

```python
        sampler = AdaptivePresenceSampler(
            video_path=_Path(ctx.video_path),
            reader_backend="auto",
            device="auto",
            presence_weights_path=weights if weights.exists() else None,
            presence_threshold=ctx.presence_threshold,
            target_yolo_fps=ctx.target_yolo_fps,
        )
```

- [ ] **Step 2: Add `target_yolo_fps` to telemetry attrs in `workers.py`**

In `src/card_capture/workers.py`, the telemetry attrs tuple (lines 478-487) reads:

```python
        for attr in (
            "last_scan_frame_count",
            "last_presence_window_count",
            "last_selected_frame_count",
            "last_score_threshold",
            "last_fallback_used",
            "last_inter_window_gaps_frames",
            "last_source_fps",
            "last_valley_splits",
        ):
```

Add `"target_yolo_fps"` to the tuple:

```python
        for attr in (
            "last_scan_frame_count",
            "last_presence_window_count",
            "last_selected_frame_count",
            "last_score_threshold",
            "last_fallback_used",
            "last_inter_window_gaps_frames",
            "last_source_fps",
            "last_valley_splits",
            "target_yolo_fps",
        ):
```

- [ ] **Step 3: Run tests**

```bash
python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py 2>&1 | tail -20
```

Expected: same pass/fail counts as after Task 3.

- [ ] **Step 4: Commit**

```bash
git add pipeline/steps/detect.py src/card_capture/workers.py
git commit -m "feat(detect): wire target_yolo_fps to sampler; add to telemetry"
```

---

### Task 5: Verify end-to-end with a real video

**Files:** None modified — observation only.

- [ ] **Step 1: Run the pipeline on a known video**

```bash
python3 -m pipeline.card_capture_flow --no-pylint run \
  --video card_capture_uploads/<any-existing-video> \
  --output-dir /tmp/stride_test \
  --db /tmp/stride_test/cards.sqlite
```

(Use any `.mp4` or `.MOV` already in `card_capture_uploads/`.)

- [ ] **Step 2: Check telemetry**

```bash
cat /tmp/stride_test/run_telemetry.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('scan_frames:    ', d.get('last_scan_frame_count'))
print('selected_frames:', d.get('last_selected_frame_count'))
print('detections:     ', d.get('detections'))
print('saved_cards:    ', d.get('saved_instances'))
print('target_yolo_fps:', d.get('target_yolo_fps'))
print('presence_windows:', d.get('last_presence_window_count'))
"
```

Expected: `last_selected_frame_count` is **roughly 1/3 to 1/4** of the pre-change value for the same video. `saved_instances` should match (same cards found).

- [ ] **Step 3: Confirm no cards lost**

If `saved_instances` is lower than expected, check `track_lengths` in `run_telemetry.json`. Tracks shorter than `min_track_length` (3) are dropped. If too many are dropping, raise `target_yolo_fps` to 4.0 or lower `min_track_length` temporarily to diagnose.
