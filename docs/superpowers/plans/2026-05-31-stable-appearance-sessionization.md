# Stable Appearance Sessionization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace centroid-driven card-session resets with stable DINOv2 appearance plateaus so the front-only `IMG_5922` capture yields 26 physical card sessions while preserving visually identical duplicate cards.

**Architecture:** Add a pure-NumPy two-pass `AppearanceSessionizer`: first confirm stable appearance plateaus, then conservatively suppress recurrent holder/null bridge clusters. Integrate it into `BoTSORTAdapter.assign()` using the embeddings already computed for BoxMOT, reset BoT-SORT only at retained plateau boundaries, and expose auditable telemetry. Add a narrow runtime worker guard for tracker execution so this feature does not introduce additional main-thread PyTorch calls; broader staged-runtime GPU-boundary cleanup remains separate architectural debt.

**Tech Stack:** Python 3, dataclasses, NumPy, pytest, BoT-SORT/BoxMOT, existing DINOv2 embeddings, existing stage-metrics DAL.

**Approved Spec:** `docs/superpowers/specs/2026-05-31-stable-appearance-sessionization-design.md`

**Source-Control Constraint:** The worktree is already dirty. Do not stage or commit files unless the user explicitly requests it. The commit steps normally required by `superpowers:writing-plans` are intentionally omitted.

---

## Task 1: Add Pure-NumPy Plateau Formation

**Files:**
- Create: `src/card_capture/tracking/appearance_sessionizer.py`
- Create: `tests/test_appearance_sessionizer.py`

- [ ] **Step 1: Write failing tests for stable plateau formation**

Create `tests/test_appearance_sessionizer.py` with helpers that use normalized,
orthogonal NumPy vectors. Cover direct in-place replacement, isolated transition
noise, and confirmation length:

```python
from __future__ import annotations

import numpy as np

from card_capture.tracking.appearance_sessionizer import (
    AppearanceObservation,
    AppearanceSessionizer,
)


def _unit(*values: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)


CARD_A = _unit(1.0, 0.0, 0.0, 0.0)
CARD_B = _unit(0.0, 1.0, 0.0, 0.0)
NOISE = _unit(0.0, 0.0, 1.0, 0.0)


def _obs(frame: int, emb: np.ndarray, novelty: float = 0.12) -> AppearanceObservation:
    return AppearanceObservation(
        frame_index=frame,
        detection_id=frame,
        embedding=emb,
        novelty_score=novelty,
    )


def test_direct_front_to_front_replacement_forms_two_plateaus():
    result = AppearanceSessionizer(confirm_frames=3).sessionize([
        _obs(0, CARD_A), _obs(1, CARD_A), _obs(2, CARD_A),
        _obs(3, CARD_B), _obs(4, CARD_B), _obs(5, CARD_B),
    ])
    assert [p.frame_range for p in result.retained_plateaus] == [(0, 2), (3, 5)]
    assert result.boundary_frame_indices == [3]


def test_isolated_transition_noise_does_not_create_plateau():
    result = AppearanceSessionizer(confirm_frames=3).sessionize([
        _obs(0, CARD_A), _obs(1, CARD_A), _obs(2, CARD_A),
        _obs(3, NOISE),
        _obs(4, CARD_B), _obs(5, CARD_B), _obs(6, CARD_B),
    ])
    assert [p.frame_range for p in result.retained_plateaus] == [(0, 2), (4, 6)]
    assert result.raw_jump_count == 2


def test_unconfirmed_tail_is_not_emitted_as_physical_card():
    result = AppearanceSessionizer(confirm_frames=3).sessionize([
        _obs(0, CARD_A), _obs(1, CARD_A), _obs(2, CARD_A),
        _obs(3, CARD_B), _obs(4, CARD_B),
    ])
    assert [p.frame_range for p in result.retained_plateaus] == [(0, 2)]
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
pytest tests/test_appearance_sessionizer.py -q
```

Expected: collection fails because `card_capture.tracking.appearance_sessionizer`
does not exist.

- [ ] **Step 3: Implement the observation, plateau, result, and first-pass API**

Create `src/card_capture/tracking/appearance_sessionizer.py`. Keep this module
NumPy-only: no imports from `torch`, Kornia, DINO, or BoxMOT.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return 1.0 - float(np.dot(left, right))


def _normalized_centroid(observations: list["AppearanceObservation"]) -> np.ndarray:
    centroid = np.mean(np.stack([o.embedding for o in observations]), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 1e-12:
        raise ValueError("appearance centroid has zero norm")
    return np.asarray(centroid / norm, dtype=np.float32)


@dataclass(frozen=True)
class AppearanceObservation:
    frame_index: int
    detection_id: Hashable
    embedding: np.ndarray
    novelty_score: float


@dataclass
class AppearancePlateau:
    observations: list[AppearanceObservation]
    representative: np.ndarray
    suppressed_reason: str | None = None

    @property
    def frame_range(self) -> tuple[int, int]:
        return self.observations[0].frame_index, self.observations[-1].frame_index

    @property
    def median_novelty(self) -> float:
        return float(np.median([o.novelty_score for o in self.observations]))


@dataclass(frozen=True)
class AppearanceSessionizationResult:
    retained_plateaus: list[AppearancePlateau]
    suppressed_plateaus: list[AppearancePlateau]
    raw_jump_count: int
    boundary_frame_indices: list[int]
    frame_to_session_id: dict[int, int]

    def metrics(self) -> dict[str, object]:
        return {
            "appearance_raw_jumps": self.raw_jump_count,
            "appearance_plateaus_confirmed": len(self.retained_plateaus) + len(self.suppressed_plateaus),
            "appearance_bridge_plateaus_suppressed": len(self.suppressed_plateaus),
            "appearance_presentations_retained": len(self.retained_plateaus),
            "appearance_boundary_frames": list(self.boundary_frame_indices),
        }


@dataclass
class AppearanceSessionizer:
    same_threshold: float = 0.15
    change_threshold: float = 0.30
    confirm_frames: int = 3
    bridge_min_occurrences: int = 3
    bridge_position_ratio: float = 0.80
    bridge_neighbor_change_ratio: float = 0.80
    bridge_novelty_margin: float = 0.05
    bridge_max_length_ratio: float = 0.75

    def sessionize(
        self,
        observations: list[AppearanceObservation],
    ) -> AppearanceSessionizationResult:
        stable = self._form_stable_plateaus(observations)
        retained, suppressed = self._suppress_bridge_plateaus(stable)
        frame_to_session_id = {
            observation.frame_index: session_id
            for session_id, plateau in enumerate(retained)
            for observation in plateau.observations
        }
        return AppearanceSessionizationResult(
            retained_plateaus=retained,
            suppressed_plateaus=suppressed,
            raw_jump_count=self._raw_jump_count(observations),
            boundary_frame_indices=[p.frame_range[0] for p in retained[1:]],
            frame_to_session_id=frame_to_session_id,
        )
```

Implement `_raw_jump_count()` by counting adjacent observations with distance
greater than `change_threshold`.

Implement `_form_stable_plateaus()` with an active buffer and a pending buffer:

1. Append an observation to the active buffer if its distance to the active
   centroid is at most `same_threshold`.
2. Otherwise append it to the pending buffer if it matches the pending
   centroid; replace the pending buffer if it does not.
3. When the pending buffer reaches `confirm_frames`, emit the previous active
   buffer if it reached `confirm_frames`, promote pending to active, and clear
   pending.
4. At end-of-input, emit active only if it reached `confirm_frames`.
5. Build each `AppearancePlateau.representative` with `_normalized_centroid()`.

For this task, implement `_suppress_bridge_plateaus()` as:

```python
def _suppress_bridge_plateaus(
    self,
    plateaus: list[AppearancePlateau],
) -> tuple[list[AppearancePlateau], list[AppearancePlateau]]:
    return plateaus, []
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_appearance_sessionizer.py -q
```

Expected: `3 passed`.

## Task 2: Add Conservative Holder/Null Bridge Suppression

**Files:**
- Modify: `src/card_capture/tracking/appearance_sessionizer.py`
- Modify: `tests/test_appearance_sessionizer.py`

- [ ] **Step 1: Add failing tests for recurring bridges and physical duplicates**

Append:

```python
HOLDER = _unit(0.0, 0.0, 0.0, 1.0)


def _plateau(start: int, emb: np.ndarray, novelty: float, length: int = 3):
    return [_obs(start + offset, emb, novelty) for offset in range(length)]


def test_recurrent_short_high_novelty_bridge_is_suppressed():
    observations = (
        _plateau(0, CARD_A, 0.12, 5)
        + _plateau(10, HOLDER, 0.26, 3)
        + _plateau(20, CARD_B, 0.13, 5)
        + _plateau(30, HOLDER, 0.27, 3)
        + _plateau(40, NOISE, 0.14, 5)
        + _plateau(50, HOLDER, 0.25, 3)
        + _plateau(60, CARD_A, 0.11, 5)
    )
    result = AppearanceSessionizer(confirm_frames=3).sessionize(observations)
    assert len(result.suppressed_plateaus) == 3
    assert [p.frame_range for p in result.retained_plateaus] == [
        (0, 4), (20, 24), (40, 44), (60, 64),
    ]


def test_repeated_visual_duplicate_cards_remain_distinct_physical_sessions():
    observations = (
        _plateau(0, CARD_A, 0.12, 5)
        + _plateau(10, CARD_B, 0.13, 5)
        + _plateau(20, CARD_A, 0.12, 5)
    )
    result = AppearanceSessionizer(confirm_frames=3).sessionize(observations)
    assert [p.frame_range for p in result.retained_plateaus] == [
        (0, 4), (10, 14), (20, 24),
    ]


def test_recurrence_without_bridge_support_is_retained():
    observations = (
        _plateau(0, CARD_A, 0.12, 5)
        + _plateau(10, CARD_B, 0.13, 5)
        + _plateau(20, CARD_A, 0.12, 5)
        + _plateau(30, NOISE, 0.14, 5)
        + _plateau(40, CARD_A, 0.12, 5)
    )
    result = AppearanceSessionizer(confirm_frames=3).sessionize(observations)
    assert len(result.suppressed_plateaus) == 0
    assert len(result.retained_plateaus) == 5
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
pytest tests/test_appearance_sessionizer.py -q
```

Expected: the recurring holder suppression test fails because no plateau is
suppressed.

- [ ] **Step 3: Implement bridge clustering and conservative suppression**

In `AppearanceSessionizer._suppress_bridge_plateaus()`:

1. Greedily cluster plateau representatives with cosine distance at most
   `same_threshold`.
2. For each cluster with at least `bridge_min_occurrences`, count occurrences
   that are interior plateaus.
3. Require `interior_count / occurrence_count >= bridge_position_ratio`.
4. For interior occurrences, compare the representative-cluster IDs of the
   left and right neighbors. Require the fraction with distinct neighbor IDs to
   be at least `bridge_neighbor_change_ratio`.
5. Compute median novelty and median length for the candidate cluster and its
   neighboring plateaus.
6. Suppress only if either:
   - candidate median novelty exceeds neighbor median novelty by at least
     `bridge_novelty_margin`; or
   - candidate median length is at most `bridge_max_length_ratio` times neighbor
     median length.
7. Set `plateau.suppressed_reason = "recurrent_bridge"` for suppressed
   occurrences.
8. Preserve retained plateau order. Never merge two retained plateaus when their
   representatives match.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_appearance_sessionizer.py -q
```

Expected: `6 passed`.

## Task 3: Add Explicit Configuration Knobs

**Files:**
- Modify: `src/card_capture/config.py`
- Modify: `card_capture_config.json`
- Modify: `tests/test_config_to_request_dict.py`

- [ ] **Step 1: Add failing request-config assertions**

Extend `test_to_request_config_includes_all_back_half_fields()`:

```python
    assert d["appearance_same_threshold"] == 0.15
    assert d["appearance_change_threshold"] == 0.30
    assert d["appearance_confirm_frames"] == 3
    assert d["bridge_min_occurrences"] == 3
    assert d["bridge_position_ratio"] == 0.80
    assert d["bridge_neighbor_change_ratio"] == 0.80
    assert d["bridge_novelty_margin"] == 0.05
    assert d["bridge_max_length_ratio"] == 0.75
```

- [ ] **Step 2: Run config test and verify failure**

Run:

```bash
pytest tests/test_config_to_request_dict.py::test_to_request_config_includes_all_back_half_fields -q
```

Expected: failure on missing `appearance_same_threshold`.

- [ ] **Step 3: Add dataclass defaults and serialization**

Add the eight settings to the tracking section of `PipelineConfig`, emit them
from `to_request_config()`, and add the same values to `card_capture_config.json`
next to the existing centroid settings.

Keep `centroid_jump_ratio` and `centroid_jump_frames` for compatibility and
telemetry. They stop being authoritative reset settings in Task 5.

- [ ] **Step 4: Run config tests**

Run:

```bash
pytest tests/test_config_to_request_dict.py -q
```

Expected: all tests pass.

## Task 4: Add a Narrow Guarded Runtime Worker for Tracker Inference

**Files:**
- Create: `src/card_capture/pipeline/runtime_worker.py`
- Create: `tests/pipeline/test_runtime_worker.py`
- Modify: `src/card_capture/pipeline/runtime_local.py`
- Modify: `src/card_capture/pipeline/stages/track.py`
- Modify: `tests/pipeline/stages/test_stage_metrics_emitted.py`

- [ ] **Step 1: Write a failing worker-thread test**

Create `tests/pipeline/test_runtime_worker.py`:

```python
from __future__ import annotations

import threading

from card_capture.pipeline.runtime_worker import RuntimeWorker


def test_runtime_worker_executes_callable_on_guarded_worker_thread():
    caller = threading.get_ident()
    worker = RuntimeWorker()
    worker.start()
    try:
        worker_ident = worker.call(threading.get_ident)
    finally:
        worker.stop()
    assert worker_ident != caller
```

- [ ] **Step 2: Run the worker test and verify failure**

Run:

```bash
pytest tests/pipeline/test_runtime_worker.py -q
```

Expected: collection fails because `card_capture.pipeline.runtime_worker` does
not exist.

- [ ] **Step 3: Implement `RuntimeWorker`**

Create a single-thread queue executor whose thread target is explicitly named
`_worker`. `call()` must propagate return values and exceptions to the caller.
Use `queue.Queue`, `threading.Thread`, and a small result queue per job. Do not
import Torch or perform model work in this module.

Expose:

```python
class RuntimeWorker:
    def start(self) -> None: ...
    def call(self, fn, /, *args, **kwargs): ...
    def stop(self) -> None: ...
    def _worker(self) -> None: ...
```

- [ ] **Step 4: Wire the runtime worker lifecycle**

In `LocalPipelineRuntime.run()`:

1. Create and start one `RuntimeWorker` before stage execution.
2. Add it to `state["runtime_worker"]`.
3. Stop it in `finally` before stopping the DAL writer.

In `pipeline/stages/track.py`, route only `tracker.assign()` through the worker:

```python
    runtime_worker = state.get("runtime_worker")
    if runtime_worker is None:
        track_states = tracker.assign(detections, frames)
    else:
        track_states = runtime_worker.call(tracker.assign, detections, frames)
```

The no-worker path stays for isolated unit tests and `cpu_debug` mocks. Real
`LocalPipelineRuntime` runs route BoT-SORT, OSNet, and tracking DINO calls
through the guarded `_worker` thread.

- [ ] **Step 5: Update the track-stage metric fixture**

Keep `tests/pipeline/stages/test_stage_metrics_emitted.py` using the no-worker
path. Its fake tracker remains synchronous and does not invoke Torch.

- [ ] **Step 6: Run focused runtime and stage tests**

Run:

```bash
pytest tests/pipeline/test_runtime_worker.py tests/pipeline/stages/test_stage_metrics_emitted.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Record the bounded scope**

Add a short comment above `state["runtime_worker"]` in `runtime_local.py`:

```python
# Tracker inference is guarded here. Refinement GPU-boundary decomposition is
# separate architectural debt; do not add new main-thread Torch calls.
```

This task prevents the sessionizer integration from deepening the current
boundary violation. It does not claim to finish the broader v5.5 runtime
boundary refactor.

## Task 5: Integrate Appearance Sessionization into BoT-SORT

**Files:**
- Modify: `src/card_capture/tracking/botsort_adapter.py`
- Modify: `src/card_capture/pipeline/stages/track.py`
- Create: `tests/test_botsort_appearance_session_reset.py`
- Modify: `tests/test_botsort_centroid_reset.py`
- Modify: `tests/test_botsort_reid_embs.py`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/test_botsort_appearance_session_reset.py` using the same
`_FakeBoTSORT`, frame, and candidate patterns as
`tests/test_botsort_centroid_reset.py`. Patch
`BoTSORTAdapter._embed_candidates()` to return deterministic normalized arrays
per frame.

Add:

```python
def test_direct_appearance_replacement_resets_once_after_confirmation():
    ...
    assert adapter.last_reset_count == 1
    assert sorted({track.session_id for track in tracks}) == [0, 1]


def test_recurrent_holder_plateaus_are_not_emitted_as_sessions():
    ...
    assert adapter.sessionization_metrics["appearance_bridge_plateaus_suppressed"] == 3
    assert adapter.sessionization_metrics["appearance_presentations_retained"] == 4


def test_identical_fronts_in_distinct_plateaus_remain_distinct_sessions():
    ...
    assert sorted({track.session_id for track in tracks}) == [0, 1, 2]
```

Use `min_track_length=1` for adapter tests so assertions isolate sessionization
instead of downstream track confirmation.

- [ ] **Step 2: Replace the centroid-reset assertion**

Rename `test_centroid_jump_triggers_reset_and_new_session()` in
`tests/test_botsort_centroid_reset.py` to
`test_centroid_jump_is_telemetry_only()`.

Keep the centroid movement input and change assertions to:

```python
    assert adapter.centroid_jump_count == 1
    assert adapter.last_reset_count == 0
    assert sorted({t.session_id for t in tracks}) == [0]
```

- [ ] **Step 3: Run adapter tests and verify failure**

Run:

```bash
pytest tests/test_botsort_appearance_session_reset.py tests/test_botsort_centroid_reset.py -q
```

Expected: failures because appearance sessionization and
`centroid_jump_count` are not integrated.

- [ ] **Step 4: Integrate the sessionizer without a second embedding pass**

In `BoTSORTAdapter.__init__()`:

1. Accept the eight appearance/bridge configuration values from Task 3.
2. Construct one `AppearanceSessionizer`.
3. Add `centroid_jump_count = 0`.
4. Add `sessionization_metrics: dict[str, object] = {}`.

In `assign()`:

1. Group detections and frame images as today.
2. Compute `_embed_candidates(frame_candidates, frame_img)` exactly once per
   frame and store the result in `embeddings_by_frame`.
3. Select the top-confidence candidate for each frame and pair it with the
   corresponding row from `embeddings_by_frame`.
4. Build `AppearanceObservation` objects and call
   `self._appearance_sessionizer.sessionize(observations)`.
5. Store `result.metrics()` in `self.sessionization_metrics`.
6. Iterate retained plateaus in order. Before each retained plateau after the
   first, call `reset()` and increment `_session_id`.
7. For each observation in the retained plateau, call `process()` with the
   precomputed full candidate embedding array for that frame.
8. Skip suppressed bridge frames and unconfirmed transition frames.
9. If no plateau is retained because embeddings are unavailable, fail open:
   process all original frames in one session with existing BoxMOT behavior.
10. Continue feeding centroid observations, but increment
    `centroid_jump_count` instead of resetting the tracker.

Do not call `_embed_candidates()` from `process()` when precomputed embeddings
were supplied. This preserves one embedding pass per frame.

- [ ] **Step 5: Pass configuration from the track stage**

In `pipeline/stages/track.py`, add the Task 3 settings to the BoT-SORT kwargs:

```python
        kwargs["appearance_same_threshold"] = cfg.get("appearance_same_threshold", 0.15)
        kwargs["appearance_change_threshold"] = cfg.get("appearance_change_threshold", 0.30)
        kwargs["appearance_confirm_frames"] = cfg.get("appearance_confirm_frames", 3)
        kwargs["bridge_min_occurrences"] = cfg.get("bridge_min_occurrences", 3)
        kwargs["bridge_position_ratio"] = cfg.get("bridge_position_ratio", 0.80)
        kwargs["bridge_neighbor_change_ratio"] = cfg.get("bridge_neighbor_change_ratio", 0.80)
        kwargs["bridge_novelty_margin"] = cfg.get("bridge_novelty_margin", 0.05)
        kwargs["bridge_max_length_ratio"] = cfg.get("bridge_max_length_ratio", 0.75)
```

- [ ] **Step 6: Lock down one embedding pass**

Extend `tests/test_botsort_reid_embs.py` with a counting stub embedder and assert
that `assign()` calls `embed_array()` once per candidate, not once during
sessionization and again during `process()`.

- [ ] **Step 7: Run focused adapter tests**

Run:

```bash
pytest \
  tests/test_botsort_adapter.py \
  tests/test_botsort_appearance_thresh.py \
  tests/test_botsort_reid_embs.py \
  tests/test_botsort_centroid_reset.py \
  tests/test_botsort_appearance_session_reset.py \
  tests/test_trackstate_session_id.py \
  -q
```

Expected: all tests pass.

## Task 6: Emit Audit Telemetry and Lock Down Physical-Instance Preservation

**Files:**
- Modify: `src/card_capture/pipeline/stages/track.py`
- Modify: `tests/pipeline/stages/test_stage_metrics_emitted.py`
- Modify: `tests/pipeline/stages/test_dedup_stage.py`
- Modify: `tests/pipeline/stages/test_store_stage.py`

- [ ] **Step 1: Add failing telemetry assertion**

Update the fake tracker in
`tests/pipeline/stages/test_stage_metrics_emitted.py::test_track_emits_stage_metrics`
with:

```python
        sessionization_metrics = {
            "appearance_raw_jumps": 4,
            "appearance_plateaus_confirmed": 3,
            "appearance_bridge_plateaus_suppressed": 1,
            "appearance_presentations_retained": 2,
            "appearance_boundary_frames": [20],
        }
```

Assert those metrics are included alongside `tracks_final` and `tracks_data`.

- [ ] **Step 2: Add physical-duplicate dedup invariant**

Extend `tests/pipeline/stages/test_dedup_stage.py`:

```python
def test_intra_run_visual_duplicates_remain_in_final_cards():
    state = {
        "video_id": 1,
        "fused_canonicals": [
            {"instance_id": "physical-a", "reid_embedding": np.array([1.0, 0.0])},
            {"instance_id": "physical-b", "reid_embedding": np.array([1.0, 0.0])},
        ],
        "repos": {},
    }
    dedup_stage.run(state, telemetry=MagicMock())
    assert state["dedup_groups"][0]["duplicate_instance_ids"] == ["physical-b"]
    assert [card["instance_id"] for card in state["final_cards"]] == [
        "physical-a", "physical-b",
    ]
```

This keeps current metadata grouping behavior but prevents later code from
silently collapsing physical output instances.

- [ ] **Step 3: Add store-stage invariant**

Extend `tests/pipeline/stages/test_store_stage.py` with a fixture containing two
fused physical instances and one intra-run dedup link. Assert that store writes
and returns two card rows.

- [ ] **Step 4: Merge session metrics in the track stage**

Change the `emit_stage_metrics()` call in `pipeline/stages/track.py`:

```python
    metrics = {"tracks_final": len(track_states), "tracks_data": len(tracks_data)}
    metrics.update(getattr(tracker, "sessionization_metrics", {}))
    emit_stage_metrics(state, stage="track", metrics=metrics)
```

- [ ] **Step 5: Run focused telemetry, dedup, and store tests**

Run:

```bash
pytest \
  tests/pipeline/stages/test_stage_metrics_emitted.py \
  tests/pipeline/stages/test_dedup_stage.py \
  tests/pipeline/stages/test_store_stage.py \
  -q
```

Expected: all tests pass.

## Task 7: Reuse the Sessionizer in the Diagnostic and Verify End-to-End

**Files:**
- Modify: `scripts/diag_swap_signals.py`
- Modify only if needed after measured evidence: `src/card_capture/tracking/appearance_sessionizer.py`

- [ ] **Step 1: Replace diagnostic-only plateau logic with production sessionizer**

Keep the existing CSV export and signal summaries. Replace the ad hoc stable
segment clustering block with:

```python
from card_capture.tracking.appearance_sessionizer import (
    AppearanceObservation,
    AppearanceSessionizer,
)

observations = [
    AppearanceObservation(
        frame_index=fi,
        detection_id=fi,
        embedding=embedding_by_frame[fi],
        novelty_score=nov,
    )
    for fi, _gap, nov, _app in rows
    if fi in embedding_by_frame
]
result = AppearanceSessionizer().sessionize(observations)
print(f"retained_presentations={len(result.retained_plateaus)}")
print(f"suppressed_bridges={len(result.suppressed_plateaus)}")
print(f"boundary_frames={result.boundary_frame_indices}")
```

- [ ] **Step 2: Run non-hardware unit verification**

Run:

```bash
pytest \
  tests/test_appearance_sessionizer.py \
  tests/test_botsort_adapter.py \
  tests/test_botsort_appearance_thresh.py \
  tests/test_botsort_reid_embs.py \
  tests/test_botsort_centroid_reset.py \
  tests/test_botsort_appearance_session_reset.py \
  tests/test_trackstate_session_id.py \
  tests/test_config_to_request_dict.py \
  tests/pipeline/test_runtime_worker.py \
  tests/pipeline/stages/test_stage_metrics_emitted.py \
  tests/pipeline/stages/test_dedup_stage.py \
  tests/pipeline/stages/test_store_stage.py \
  -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run the required project suite**

Run:

```bash
pytest -m "not quarantine"
```

Expected: suite passes. If existing unrelated dirty-worktree failures remain,
record them with exact test names and do not alter unrelated user work.

- [ ] **Step 4: Run the enriched diagnostic manually on local hardware**

Per `AGENTS.md`, run performance-sensitive and final processing commands
manually in a local terminal:

```bash
.venv/bin/python scripts/diag_swap_signals.py > out/diag_swap.log 2>&1
```

Expected summary:

```text
retained_presentations=26
suppressed_bridges=18
```

- [ ] **Step 5: Run the full pipeline manually on local hardware**

Run:

```bash
.venv/bin/python -m card_capture.cli process \
  card_capture_uploads/bc827fce3adf4b1ea08ea8e0dec47fb8_IMG_5922.MOV \
  --output-dir out/manual_img5922_sessionization \
  --db out/manual_img5922_sessionization/cards.sqlite \
  --config card_capture_config.json \
  --run-id img5922-sessionization
```

Verify:

- Final physical card count is `26`.
- Every output is a front crop.
- Visually identical physical duplicates remain separate output rows.
- Tracker telemetry reports `26` retained presentations and `18` suppressed
  bridges for this capture.
- GPU/MPS timing is measured manually; do not claim performance from restricted
  agent execution.

## Task 8: Review Scope and Dirty Worktree Before Handoff

**Files:**
- Inspect only: all modified files

- [ ] **Step 1: Review diff scope**

Run:

```bash
git status --short
git diff --stat
```

Expected: sessionization changes are limited to the files listed in this plan.
Existing unrelated dirty-worktree files remain untouched.

- [ ] **Step 2: Verify no artifacts are staged**

Run:

```bash
git diff --cached --stat
```

Expected: no staged changes unless the user explicitly requested staging.

- [ ] **Step 3: Summarize residual architecture debt**

Report that Task 4 guards tracker inference introduced by this feature. The
existing staged runtime still needs a separate decomposition pass to move all
pre-existing refinement Torch/Kornia work into guarded worker calls while
keeping CPU metadata work on the main thread.
