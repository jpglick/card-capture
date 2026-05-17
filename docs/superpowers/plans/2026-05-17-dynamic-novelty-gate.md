# Dynamic Novelty Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically activate the background-novelty track-pruning gate only when the per-video novelty score distribution indicates the background model is discriminating, filtering acrylic-stand false positives on stand-style videos without breaking hand-held videos.

**Architecture:** The novelty step already populates `ctx.observed_novelty_scores` (a list of per-detection novelty scores, 0–1). The score step reads those scores, applies a three-condition distribution test to decide if the gate should fire, then prunes tracks whose median novelty falls below `ctx.novelty_floor` (default 0.30) when the gate is active. Zero changes to intermediate dataclasses.

**Tech Stack:** Python, NumPy, existing Metaflow pipeline (`pipeline/steps/score.py`, `pipeline/steps/start.py`, `src/card_capture/config.py`).

**Spec:** `docs/superpowers/specs/2026-05-17-dynamic-novelty-gate-design.md`

---

### Task 1: Add `novelty_floor` to config and RunContext

**Files:**
- Modify: `src/card_capture/config.py:35` (after `target_yolo_fps`)
- Modify: `pipeline/steps/start.py:51` (after `target_yolo_fps` field) and `pipeline/steps/start.py:153` (after `target_yolo_fps=cfg.target_yolo_fps` in `init_run`)

- [ ] **Step 1: Add `novelty_floor` to `PipelineConfig`**

In `src/card_capture/config.py`, after line 35 (`target_yolo_fps: float = 3.0`), add:

```python
    target_yolo_fps: float = 3.0
    novelty_floor: float = 0.30
```

- [ ] **Step 2: Add `novelty_floor` to `RunContext`**

In `pipeline/steps/start.py`, after line 51 (`target_yolo_fps: float = 3.0`), add:

```python
    target_yolo_fps: float = 3.0
    novelty_floor: float = 0.30
```

- [ ] **Step 3: Wire `novelty_floor` in `init_run`**

In `pipeline/steps/start.py`, after line 153 (`target_yolo_fps=cfg.target_yolo_fps,`), add:

```python
        target_yolo_fps=cfg.target_yolo_fps,
        novelty_floor=cfg.novelty_floor,
```

- [ ] **Step 4: Run tests to confirm no breakage**

```bash
python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py 2>&1 | tail -10
```

Expected: same pass/fail counts as before (pre-existing failures documented in CLAUDE.md are not regressions).

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/config.py pipeline/steps/start.py
git commit -m "feat(config): add novelty_floor field (default 0.30)"
```

---

### Task 2: Write failing tests for `_novelty_gate_useful`

**Files:**
- Create: `tests/pipeline/test_score_novelty_gate.py`

- [ ] **Step 1: Create the test file**

Create `tests/pipeline/test_score_novelty_gate.py` with this content:

```python
import pytest


def _gate(scores):
    from pipeline.steps.score import _novelty_gate_useful
    return _novelty_gate_useful(scores)


def test_bimodal_distribution_activates_gate():
    """Stand-style video: low-novelty stand detections + high-novelty card detections."""
    scores = [0.05, 0.08, 0.85, 0.90, 0.92]
    assert _gate(scores) is True


def test_all_high_scores_disables_gate():
    """Hand-held video: everything is novel vs background — std too low."""
    scores = [0.82, 0.88, 0.91, 0.87, 0.85]
    assert _gate(scores) is False


def test_too_few_samples_disables_gate():
    """Fewer than 5 detections — not enough data to judge distribution."""
    scores = [0.05, 0.90]
    assert _gate(scores) is False


def test_high_std_but_min_not_low_enough_disables_gate():
    """Spread exists but nothing scores below 0.35 — no background-like detections."""
    scores = [0.40, 0.90, 0.91, 0.40, 0.88]
    assert _gate(scores) is False


def test_empty_scores_disables_gate():
    """No detections at all — gate must not fire."""
    assert _gate([]) is False


def test_exactly_five_samples_bimodal_activates():
    """Boundary: exactly 5 samples with bimodal distribution."""
    scores = [0.10, 0.12, 0.80, 0.85, 0.88]
    assert _gate(scores) is True
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python3 -m pytest tests/pipeline/test_score_novelty_gate.py -v 2>&1 | tail -15
```

Expected: all 6 tests FAIL with `ImportError: cannot import name '_novelty_gate_useful' from 'pipeline.steps.score'`

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/pipeline/test_score_novelty_gate.py
git commit -m "test(score): failing tests for dynamic novelty gate helper"
```

---

### Task 3: Implement `_novelty_gate_useful` and update score step

**Files:**
- Modify: `pipeline/steps/score.py` (add helper, update `run()`)

- [ ] **Step 1: Add `_novelty_gate_useful` helper**

In `pipeline/steps/score.py`, add the helper function after the imports and before the `ScoreOutput` dataclass (after line 14):

```python
def _novelty_gate_useful(scores: list) -> bool:
    """Return True only when the background model discriminates well enough to prune.

    Requires all three:
    - At least 5 detections (enough data)
    - std > 0.15 (model is actually separating high from low novelty)
    - min < 0.35 (at least some detections look like background)
    """
    if len(scores) < 5:
        return False
    import numpy as np
    arr = np.array(scores, dtype=np.float32)
    return float(arr.std()) > 0.15 and float(arr.min()) < 0.35
```

- [ ] **Step 2: Replace the entire `run()` function**

In `pipeline/steps/score.py`, replace the entire `run()` function (lines 33–97) with:

```python
def run(ctx: RunContext, refine_out: RefineOutput) -> ScoreOutput:
    """Prune empty-workspace tracks and attach quality scores.

    Analyses ctx.observed_novelty_scores distribution to decide if the
    background model is discriminating enough to use as a gate. When active,
    prunes any track whose median novelty score falls below ctx.novelty_floor.
    When the gate is not useful (e.g. hand-held video), no tracks are pruned.

    Args:
        ctx:        RunContext from the start step.
        refine_out: Output from the refine step.

    Returns:
        ``ScoreOutput`` with scored / pruned track lists.
    """
    import numpy as np

    bg_model = None
    if refine_out.bg_model_path and Path(refine_out.bg_model_path).exists():
        from card_capture.presence.background_novelty import BackgroundModel
        mean_bgr = np.load(refine_out.bg_model_path)
        bg_model = BackgroundModel.__new__(BackgroundModel)
        bg_model.mean_bgr = mean_bgr

    gate_useful = _novelty_gate_useful(ctx.observed_novelty_scores)
    threshold = ctx.novelty_floor if gate_useful else -1.0

    scored_tracks: List[Dict[str, Any]] = []
    pruned_instance_ids: List[str] = []

    for track_dict in refine_out.refined_tracks:
        frame_entries = track_dict.get("frame_entries", [])

        novelty_scores = [
            float(fe.get("novelty_score", 1.0))
            for fe in frame_entries
        ]
        median_novelty = float(np.median(novelty_scores)) if novelty_scores else 1.0

        should_prune = (bg_model is not None) and gate_useful and (median_novelty < threshold)

        track_out = dict(track_dict)
        track_out["pruned"] = should_prune
        track_out["median_novelty"] = median_novelty

        scored_tracks.append(track_out)
        if should_prune:
            pruned_instance_ids.append(track_dict["instance_id"])

    active_count = sum(1 for t in scored_tracks if not t["pruned"])
    print(
        f"[Stage: Score] | {len(scored_tracks)} tracks scored"
        f" | {len(pruned_instance_ids)} pruned | {active_count} active"
        f" | novelty_gate={'on' if gate_useful else 'off'}"
        f" | threshold={threshold:.2f}"
    )

    return ScoreOutput(
        scored_tracks=scored_tracks,
        pruned_instance_ids=pruned_instance_ids,
        tracker_events=refine_out.tracker_events,
        detection_rows=refine_out.detection_rows,
        sampler_telemetry=refine_out.sampler_telemetry,
        bg_model_path=refine_out.bg_model_path,
        accepted_frame_presence=refine_out.accepted_frame_presence,
        frame_count=refine_out.frame_count,
        accepted_frame_count=refine_out.accepted_frame_count,
        video_id=refine_out.video_id,
    )
```

- [ ] **Step 3: Run the new tests**

```bash
python3 -m pytest tests/pipeline/test_score_novelty_gate.py -v 2>&1 | tail -15
```

Expected: all 6 tests PASS.

- [ ] **Step 4: Run the full test suite**

```bash
python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py 2>&1 | tail -10
```

Expected: same pass/fail counts as after Task 1 (no new failures).

- [ ] **Step 5: Commit**

```bash
git add pipeline/steps/score.py
git commit -m "feat(score): dynamic novelty gate — activates only when bg model discriminates

Replaces hardcoded < 0.0 threshold (never fired) with a distribution-based
decision: gate activates when novelty scores have std > 0.15 and min < 0.35
(bimodal distribution indicating the background model can separate stand
detections from real card detections). Falls back to disabled for hand-held
videos where all scores cluster high."
```
