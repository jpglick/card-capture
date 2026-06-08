# Refine Card-Region Crop Pre-Pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `refine` from holding the full 4K frame buffer resident while it GPU-warps, by converting frames to small per-candidate card-region crops in a one-time pre-pass — eliminating the "5 GB baseline + warp spike" overlap that jetsam-kills long-video runs.

**Architecture:** Before the per-track warp loop, select the exact candidates refine will warp (top-8 per track), crop each one's quad bounding box (+margin) from its source frame as a `.copy()`, free each full frame immediately, then run the warp loop on the small crops. The warp is translation-equivariant in the source quad, so `warp(full_frame, corners) == warp(crop, corners − crop_offset)` pixel-for-pixel — outputs are unchanged. Implements Phase 1 §5.1 of `docs/superpowers/specs/2026-06-07-pipeline-memory-perf-holistic-design.md`.

**Tech Stack:** Python 3.9, NumPy, OpenCV (`PrecisionNormalizer`), Kornia/Torch (`KorniaNormalizer`), pytest. MPS-only accelerator; tests run on CPU.

---

## Scope

This plan covers **only** the refine crop pre-pass (spec §5.1). Out of scope (own future plans): novelty dict-lookup + downscaled Lab (§5.2), the resident-buffer architecture decision (Phase 2), stage pipelining (Phase 3).

**Deliberate deviation from the spec:** the spec mentions keeping a `refine_warp_chunk_size` knob. With crops the per-warp upload is ~20× smaller, so chunking is moot — omitted (YAGNI). The available-memory gate (§5.1 "single pre-pass check") is included as the final task.

## File Structure

- **Modify** `src/card_capture/stages/refine/__init__.py` — add `math` import; add pure helpers `_quad_bbox`, `_crop_and_rebase`, `_plan_crop_candidates`, `_available_memory_mb`; insert the crop pre-pass into `run()`; rewrite the two frame-reads in the warp loop to consume crops.
- **Modify** `src/card_capture/core/config.py` — add `refine_crop_margin_px` and `refine_min_available_mb` fields to `PipelineConfig` and emit them in `to_request_config()`.
- **Create** `tests/pipeline/stages/test_refine_crop_prepass.py` — keystone warp-equivalence test, helper unit tests, integration test, gate test, config-flow test.

Refine currently reads frames at `__init__.py:123` (Kornia batch build) and `__init__.py:143` (fallback normalize). Both become crop lookups.

---

## Task 1: Config knobs

**Files:**
- Modify: `src/card_capture/core/config.py`
- Test: `tests/pipeline/stages/test_refine_crop_prepass.py`

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/stages/test_refine_crop_prepass.py` with:

```python
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.core.config import PipelineConfig
from card_capture.stages import refine
from card_capture.stages.refine.cropper import PrecisionNormalizer


def test_config_exposes_crop_knobs_in_request_dict():
    cfg = PipelineConfig()
    d = cfg.to_request_config()
    assert d["refine_crop_margin_px"] == 8
    assert d["refine_min_available_mb"] == 2048.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py::test_config_exposes_crop_knobs_in_request_dict -q`
Expected: FAIL with `KeyError: 'refine_crop_margin_px'`.

- [ ] **Step 3: Add the fields and emit them**

In `src/card_capture/core/config.py`, under the `# Post-Processing` block (after `fusion_target_frames: int = 1`), add:

```python
    # Refine memory safety (16GB-box OOM guard); see refine crop pre-pass.
    refine_crop_margin_px: int = 8
    refine_min_available_mb: float = 2048.0
```

In `to_request_config()`, in the `# Refine / fusion` group (after `"kornia_device": self.device,`), add:

```python
            "refine_crop_margin_px": self.refine_crop_margin_px,
            "refine_min_available_mb": self.refine_min_available_mb,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py::test_config_exposes_crop_knobs_in_request_dict -q`
Expected: PASS.

- [ ] **Step 5: Run the existing config test to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_config_to_request_dict.py tests/test_config_back_half_fields.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/core/config.py tests/pipeline/stages/test_refine_crop_prepass.py
git commit -m "$(cat <<'EOF'
feat(refine): add refine_crop_margin_px / refine_min_available_mb config knobs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `_quad_bbox` + `_crop_and_rebase` helpers (the correctness keystone)

**Files:**
- Modify: `src/card_capture/stages/refine/__init__.py`
- Test: `tests/pipeline/stages/test_refine_crop_prepass.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/stages/test_refine_crop_prepass.py`:

```python
def test_quad_bbox_clamps_to_frame_and_applies_margin():
    # quad inside a 200x300 (w x h) frame; margin expands, clamped at 0/edge.
    corners = [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)]
    assert refine._quad_bbox(corners, 200, 300, margin=8) == (30, 42, 168, 248)


def test_quad_bbox_clamps_when_quad_touches_edge():
    corners = [(2.0, 1.0), (199.0, 0.0), (198.0, 299.0), (1.0, 298.0)]
    assert refine._quad_bbox(corners, 200, 300, margin=8) == (0, 0, 200, 300)


def test_crop_and_rebase_returns_a_copy_not_a_view():
    frame = (np.random.RandomState(1).rand(300, 200, 3) * 255).astype(np.uint8)
    corners = [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)]
    crop, _ = refine._crop_and_rebase(frame, corners, margin=8)
    assert crop.base is None  # a real copy, so freeing `frame` releases its memory


def test_crop_and_rebase_warp_equivalence():
    """warp(full, corners) is pixel-identical to warp(crop, rebased corners)."""
    frame = (np.random.RandomState(0).rand(300, 200, 3) * 255).astype(np.uint8)
    corners = [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)]
    norm = PrecisionNormalizer()

    full = norm.normalize(frame, corners, rotate_180=False)
    crop, local = refine._crop_and_rebase(frame, corners, margin=8)
    cropped = norm.normalize(crop, local, rotate_180=False)

    assert np.array_equal(full, cropped)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py -k "quad_bbox or crop_and_rebase" -q`
Expected: FAIL with `AttributeError: module 'card_capture.stages.refine' has no attribute '_quad_bbox'`.

- [ ] **Step 3: Implement the helpers**

In `src/card_capture/stages/refine/__init__.py`, change the import line `from typing import Any, Dict, List, Optional` to add `Tuple`, and add `import math`:

```python
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
```

Then add these functions just after `_frame_index_lookup` (after line 60):

```python
def _quad_bbox(corners, frame_w: int, frame_h: int, margin: int) -> Tuple[int, int, int, int]:
    """Axis-aligned bounding box of a card quad, expanded by `margin` and
    clamped to the frame. The margin gives the perspective warp interpolation
    neighbours at the quad edge."""
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    x0 = max(0, int(math.floor(min(xs))) - margin)
    y0 = max(0, int(math.floor(min(ys))) - margin)
    x1 = min(int(frame_w), int(math.ceil(max(xs))) + margin)
    y1 = min(int(frame_h), int(math.ceil(max(ys))) + margin)
    return x0, y0, x1, y1


def _crop_and_rebase(frame: np.ndarray, corners, margin: int):
    """Crop the card region out of `frame` (always a `.copy()`, never a view —
    a view would pin the full frame and defeat the memory win) and rebase the
    corners into crop-local coordinates. `warp(frame, corners)` ==
    `warp(crop, rebased)` because the perspective warp is translation-equivariant
    in the source quad."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = _quad_bbox(corners, w, h, margin)
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    crop = frame[y0:y1, x0:x1].copy()
    local = [(float(px) - x0, float(py) - y0) for px, py in corners]
    return crop, local
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py -k "quad_bbox or crop_and_rebase" -q`
Expected: PASS (4 tests). If `test_crop_and_rebase_warp_equivalence` fails by ≤1 grey level at the border, the margin is too small for that quad — it should not at margin=8 with the quad fully inside; investigate rather than relaxing the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/stages/refine/__init__.py tests/pipeline/stages/test_refine_crop_prepass.py
git commit -m "$(cat <<'EOF'
feat(refine): add _quad_bbox + _crop_and_rebase with warp-equivalence test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `_plan_crop_candidates` helper

**Files:**
- Modify: `src/card_capture/stages/refine/__init__.py`
- Test: `tests/pipeline/stages/test_refine_crop_prepass.py`

- [ ] **Step 1: Write the failing test**

Append to the test file:

```python
def _candidate(detection_id, frame_index, score):
    return {
        "detection_id": detection_id,
        "frame_index": frame_index,
        "score_total": score,
        "corners": [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    }


def test_plan_crop_candidates_keeps_top_n_per_track_keyed_by_detection():
    tracks = [
        {"candidates": [_candidate(1, 5, 0.9), _candidate(2, 6, 0.1), _candidate(3, 7, 0.5)]},
    ]
    wanted = refine._plan_crop_candidates(tracks, top_n=2)
    # Top-2 by score = detections 1 (0.9) and 3 (0.5); detection 2 dropped.
    assert set(wanted.keys()) == {1, 3}
    assert wanted[1] == (5, [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py::test_plan_crop_candidates_keeps_top_n_per_track_keyed_by_detection -q`
Expected: FAIL with `AttributeError: ... has no attribute '_plan_crop_candidates'`.

- [ ] **Step 3: Implement the helper**

In `src/card_capture/stages/refine/__init__.py`, add after `_crop_and_rebase`:

```python
def _plan_crop_candidates(tracks_data, top_n: int = 8) -> Dict[int, Tuple[int, list]]:
    """Map detection_id -> (frame_index, corners) for the top-N scored
    candidates of each track — exactly the set the warp loop will consume,
    using the same sort as the loop so the crop set matches 1:1."""
    wanted: Dict[int, Tuple[int, list]] = {}
    for track_dict in tracks_data:
        scored = sorted(
            track_dict.get("candidates", []),
            key=lambda c: c.get("score_total", 0.0), reverse=True,
        )[:top_n]
        for c in scored:
            wanted[int(c["detection_id"])] = (int(c["frame_index"]), c.get("corners") or [])
    return wanted
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py::test_plan_crop_candidates_keeps_top_n_per_track_keyed_by_detection -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/stages/refine/__init__.py tests/pipeline/stages/test_refine_crop_prepass.py
git commit -m "$(cat <<'EOF'
feat(refine): add _plan_crop_candidates (top-N per track, keyed by detection)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire the crop pre-pass into `run()` and consume crops in the warp loop

**Files:**
- Modify: `src/card_capture/stages/refine/__init__.py` (`run()`, current lines 84-149)
- Test: `tests/pipeline/stages/test_refine_crop_prepass.py`

- [ ] **Step 1: Write the failing integration test**

Append to the test file:

```python
def _frame(idx, w=200, h=300):
    img = (np.random.RandomState(idx).rand(h, w, 3) * 255).astype(np.uint8)
    fs = MagicMock()
    fs.frame_index = idx
    fs.image = img
    fs.width = w
    fs.height = h
    fs.timestamp_ms = idx * 33
    return fs


def _track(instance_id, frame_indices):
    return {
        "instance_id": instance_id,
        "track_id": 1,
        "angle": "Unknown",
        "session_id": 0,
        "first_frame_index": frame_indices[0],
        "candidates": [
            {
                "detection_id": idx * 10 + 1,
                "frame_index": idx,
                "timestamp_ms": idx * 33,
                "width": 200,
                "height": 300,
                "corners": [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)],
                "confidence": 0.9,
                "novelty_score": 1.0,
                "score_total": 0.7,
                "image_path": "",
                "triage_metrics": {},
            }
            for idx in frame_indices
        ],
    }


class _CapturingTelemetry:
    def __init__(self):
        self.samples = []

    def resource_sample(self, payload):
        self.samples.append(payload)

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _state(tracks, n_frames=20):
    request = MagicMock()
    request.config = {
        "device": "cpu", "rotate_180": False, "use_kornia": True, "kornia_device": "cpu",
        "laplacian_scan_stride": 0, "max_corner_gap_frames": 30, "corner_refinement": False,
        "fusion_target_frames": 1, "refine_crop_margin_px": 8, "refine_min_available_mb": 0.0,
    }
    return {
        "request": request,
        "sampled_frames": [_frame(i) for i in range(n_frames)],
        "tracks_data": tracks,
        "detections": [],
        "video_id": 1,
        "db_path": "/tmp/x.sqlite",
    }


def test_refine_frees_frames_in_prepass_and_still_refines():
    state = _state([_track("inst-a", [5, 10, 15])])
    tele = _CapturingTelemetry()
    refine.run(state, telemetry=tele)

    assert state.get("refined_tracks")
    assert state["refined_tracks"][0]["frame_entries"]
    assert not state.get("sampled_frames")  # raw buffer released
    cropped = [s for s in tele.samples if s.get("event") == "refine_cropped"]
    assert cropped and cropped[0]["frames_freed"] == 3  # frames 5, 10, 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py::test_refine_frees_frames_in_prepass_and_still_refines -q`
Expected: FAIL — no `refine_cropped` event emitted (`cropped` is empty), and/or `sampled_frames` still populated.

- [ ] **Step 3: Insert the pre-pass and rewrite the two frame reads**

In `src/card_capture/stages/refine/__init__.py`, replace the block that currently reads (lines 103-106):

```python
    refined_tracks: List[Dict[str, Any]] = []
    tracks_data = state.get("tracks_data") or []
    
    total_tracks = len(tracks_data)
```

with:

```python
    refined_tracks: List[Dict[str, Any]] = []
    tracks_data = state.get("tracks_data") or []
    total_tracks = len(tracks_data)

    # --- Crop pre-pass (spec 2026-06-07 Phase 1 §5.1) ----------------------
    # refine is the last consumer of the raw frames; nothing downstream reads
    # state["sampled_frames"]. Convert each top-N candidate to a small card-
    # region crop and free the full 4K frame immediately, so the warp loop
    # below never coexists with the multi-GB frame buffer.
    margin = int(config.get("refine_crop_margin_px", 8))
    wanted = _plan_crop_candidates(tracks_data, top_n=8)
    by_frame: Dict[int, List[int]] = {}
    for det_id, (fidx, _corners) in wanted.items():
        by_frame.setdefault(fidx, []).append(det_id)

    crops: Dict[int, Tuple[Optional[np.ndarray], list]] = {}
    frames_freed = 0
    for fidx, det_ids in by_frame.items():
        frame = decoded_images.get(fidx)
        for det_id in det_ids:
            _fidx, corners = wanted[det_id]
            if frame is None or not corners:
                crops[det_id] = (None, corners)
            else:
                crops[det_id] = _crop_and_rebase(frame, corners, margin)
        if decoded_images.pop(fidx, None) is not None:
            frames_freed += 1
    decoded_images.clear()
    state["sampled_frames"] = []
    frames = None
    telemetry.resource_sample(
        {"event": "refine_cropped", "crops": len(crops), "frames_freed": frames_freed}
    )
```

Then, in the warp loop, replace the Kornia batch build (current lines 119-129):

```python
        if kornia_normalizer is not None and scored_candidates:
            batch_items = []
            batch_ids = []
            for c in scored_candidates:
                raw = decoded_images.get(int(c["frame_index"]))
                if raw is None:
                    h = int(c.get("height", 10))
                    w = int(c.get("width", 10))
                    raw = np.zeros((h, w, 3), dtype=np.uint8)
                batch_items.append((raw, c["corners"]))
                batch_ids.append(int(c["detection_id"]))
```

with:

```python
        if kornia_normalizer is not None and scored_candidates:
            batch_items = []
            batch_ids = []
            for c in scored_candidates:
                det_id = int(c["detection_id"])
                crop, local_corners = crops.get(det_id, (None, c.get("corners") or []))
                if crop is None:
                    h = int(c.get("height", 10))
                    w = int(c.get("width", 10))
                    crop = np.zeros((h, w, 3), dtype=np.uint8)
                batch_items.append((crop, local_corners))
                batch_ids.append(det_id)
```

And replace the fallback read in the `frame_entries` loop (current lines 142-149):

```python
        for c in scored_candidates:
            raw = decoded_images.get(int(c["frame_index"]))
            if raw is None:
                raw = np.zeros((int(c.get("height", 10)), int(c.get("width", 10)), 3),
                               dtype=np.uint8)
            normalized = normalized_by_det.get(int(c["detection_id"]))
            if normalized is None:
                normalized = normalizer.normalize(raw, c["corners"], rotate_180=rotate_180)
```

with:

```python
        for c in scored_candidates:
            det_id = int(c["detection_id"])
            crop, local_corners = crops.get(det_id, (None, c.get("corners") or []))
            if crop is None:
                crop = np.zeros((int(c.get("height", 10)), int(c.get("width", 10)), 3),
                                dtype=np.uint8)
            normalized = normalized_by_det.get(det_id)
            if normalized is None:
                normalized = normalizer.normalize(crop, local_corners, rotate_180=rotate_180)
```

(The stored `frame_entries[...]["corners"]` and `_scored_candidate_from_dict` keep using `c["corners"]` — full-frame coordinates — unchanged, so downstream track telemetry is unaffected.)

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py::test_refine_frees_frames_in_prepass_and_still_refines -q`
Expected: PASS.

- [ ] **Step 5: Run the full new test file + adjacent refine/runtime tests**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py tests/pipeline/stages/ tests/test_gpu_warp_tensor.py -q`
Expected: PASS (no regressions in the refine stage or warp tests).

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/stages/refine/__init__.py tests/pipeline/stages/test_refine_crop_prepass.py
git commit -m "$(cat <<'EOF'
feat(refine): crop card regions in a pre-pass; free 4K frames before warps

Eliminates the resident-frame-buffer + warp-spike overlap that OOM-killed long
4K runs. Warp output is unchanged (translation-equivariant crop).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Single pre-pass available-memory gate (clean failure if already over the line)

**Files:**
- Modify: `src/card_capture/stages/refine/__init__.py`
- Test: `tests/pipeline/stages/test_refine_crop_prepass.py`

- [ ] **Step 1: Write the failing tests**

Append to the test file:

```python
def test_available_memory_mb_returns_inf_when_psutil_unavailable(monkeypatch):
    import psutil

    def _boom():
        raise RuntimeError("no psutil")

    monkeypatch.setattr(psutil, "virtual_memory", _boom)
    assert refine._available_memory_mb() == math.inf


def test_refine_aborts_cleanly_when_below_memory_floor(monkeypatch):
    monkeypatch.setattr(refine, "_available_memory_mb", lambda: 100.0)
    state = _state([_track("inst-a", [5, 10, 15])])
    state["request"].config["refine_min_available_mb"] = 4096.0
    with pytest.raises(RuntimeError, match="memory"):
        refine.run(state, telemetry=MagicMock())
    assert not state.get("sampled_frames")  # buffer released on the failure path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py -k "available_memory or aborts_cleanly" -q`
Expected: FAIL with `AttributeError: ... has no attribute '_available_memory_mb'`.

- [ ] **Step 3: Add the probe and the single gate**

In `src/card_capture/stages/refine/__init__.py`, add after `_plan_crop_candidates`:

```python
def _available_memory_mb() -> float:
    """Available physical memory in MiB — the quantity macOS jetsam acts on.
    Returns inf if psutil is unavailable so the gate never aborts spuriously."""
    try:
        import psutil
        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        return math.inf
```

Then, in `run()`, insert the gate immediately before the crop pre-pass `margin = ...` line (so it runs once, before any crop allocation):

```python
    min_available_mb = float(config.get("refine_min_available_mb", 2048.0))
    available_mb = _available_memory_mb()
    if available_mb < min_available_mb:
        decoded_images.clear()
        state["sampled_frames"] = []
        raise RuntimeError(
            f"refine aborted: only {available_mb:.0f}MB physical memory available "
            f"(< {min_available_mb:.0f}MB floor) — refusing to start 4K cropping/"
            f"warps to avoid an OS memory-pressure SIGKILL. Close other apps, "
            f"process a shorter clip, lower fast_scan_fps, or lower "
            f"refine_min_available_mb to override."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py -k "available_memory or aborts_cleanly" -q`
Expected: PASS.

- [ ] **Step 5: Run the whole new test file**

Run: `.venv/bin/python -m pytest tests/pipeline/stages/test_refine_crop_prepass.py -q`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/stages/refine/__init__.py tests/pipeline/stages/test_refine_crop_prepass.py
git commit -m "$(cat <<'EOF'
feat(refine): fail cleanly when physical memory is below floor at refine entry

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Full-suite regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the full non-quarantine suite**

Run: `.venv/bin/python -m pytest tests/ -m "not quarantine" -q`
Expected: PASS except the 4 pre-existing, unrelated failures in `tests/test_background_novelty_perf.py` (missing `BackgroundModel.lab_mean`, a separate novelty workstream). No new failures in refine/config/runtime/warp tests.

- [ ] **Step 2: If green, the plan is complete.** No commit (verification only).

---

## Self-Review

**Spec coverage (§5.1):** frame-ordered crop + free (Task 4 ✓); coordinate contract + `.copy()` (Task 2 ✓); keystone warp-equivalence test (Task 2 ✓); warp loop consumes crops on both the Kornia and `PrecisionNormalizer` paths (Task 4 ✓); `refine_cropped` telemetry replacing the parked per-track `refine_frame_buffer` (Task 4 ✓ — the parked per-track free is not on HEAD, so nothing to remove); `refine_crop_margin_px` knob (Task 1 ✓); single pre-pass memory gate (Task 5 ✓). Spec's `refine_warp_chunk_size` intentionally omitted (Scope note — moot with crops).

**Placeholder scan:** none — every code/command step is concrete.

**Type consistency:** `crops` is `Dict[int, Tuple[Optional[np.ndarray], list]]` keyed by `detection_id` everywhere (pre-pass build + both loop reads). `_plan_crop_candidates` returns `Dict[int, Tuple[int, list]]` (det_id → (frame_index, corners)); the pre-pass unpacks `(fidx, corners)` consistently. `_crop_and_rebase` returns `(np.ndarray, list-of-(float,float))`; the warp loop passes that tuple straight into `warp_canonical_batch` / `normalizer.normalize`, matching their existing `(image, corners)` signatures. Config keys `refine_crop_margin_px` / `refine_min_available_mb` match between `config.py`, `to_request_config()`, and the test `request.config` dict.
