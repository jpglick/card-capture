# Frame Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two complementary pipeline quality improvements — always scan the first seconds of video to catch cards resting on stands (Feature A), and replace temporal-stride frame selection with an in-track Laplacian sharpness scan to give the warp step the best available frames (Feature B).

**Architecture:** Feature A adds sparse opening-window frames to the sampler's decode set, bypassing the presence gate. Feature B adds a single-pass Laplacian sharpness scan in the refine step that finds the sharpest frames within each confirmed track's time range and reorders/supplements the candidates fed to the Kornia warp.

**Tech Stack:** Python, OpenCV (Laplacian variance), existing `_open_capture` / `AdaptivePresenceSampler` / refine step.

**Spec:** `docs/superpowers/specs/2026-05-17-frame-quality-improvements-design.md`

---

### Task 1: Add config fields for both features

**Files:**
- Modify: `src/card_capture/config.py:39`
- Modify: `pipeline/steps/start.py:55` (RunContext field) and `:161` (init_run wiring)
- Modify: `pipeline/steps/detect.py:165` (pass opening_scan_s to AdaptivePresenceSampler)

- [ ] **Step 1: Add three fields to `PipelineConfig`**

In `src/card_capture/config.py`, after line 39 (`stand_sharpness_max: float = 0.092`):

```python
    stand_sharpness_max: float = 0.092
    opening_scan_s: float = 2.0
    laplacian_scan_stride: int = 4
    max_corner_gap_frames: int = 15
```

- [ ] **Step 2: Add three fields to `RunContext`**

In `pipeline/steps/start.py`, after line 55 (`stand_sharpness_max: float = 0.092`):

```python
    stand_sharpness_max: float = 0.092
    opening_scan_s: float = 2.0
    laplacian_scan_stride: int = 4
    max_corner_gap_frames: int = 15
```

- [ ] **Step 3: Wire in `init_run`**

In `pipeline/steps/start.py`, after line 161 (`stand_sharpness_max=cfg.stand_sharpness_max,`):

```python
        stand_sharpness_max=cfg.stand_sharpness_max,
        opening_scan_s=cfg.opening_scan_s,
        laplacian_scan_stride=cfg.laplacian_scan_stride,
        max_corner_gap_frames=cfg.max_corner_gap_frames,
```

- [ ] **Step 4: Pass `opening_scan_s` to `AdaptivePresenceSampler` in detect.py**

In `pipeline/steps/detect.py`, in `_build_sampler_detector`, update the `AdaptivePresenceSampler(...)` constructor call:

```python
        sampler = AdaptivePresenceSampler(
            video_path=_Path(ctx.video_path),
            reader_backend="auto",
            device="auto",
            presence_weights_path=weights if weights.exists() else None,
            presence_threshold=ctx.presence_threshold,
            target_yolo_fps=ctx.target_yolo_fps,
            opening_scan_s=ctx.opening_scan_s,
        )
```

- [ ] **Step 5: Run tests to confirm no breakage**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py tests/pipeline/test_score_novelty_gate.py -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/config.py pipeline/steps/start.py pipeline/steps/detect.py
git commit -m "feat(config): add opening_scan_s, laplacian_scan_stride, max_corner_gap_frames"
```

---

### Task 2: Write failing tests for `_compute_opening_indices`

**Files:**
- Test: `tests/test_sampler_fast_scan.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_sampler_fast_scan.py`:

```python
def test_opening_scan_indices_two_seconds():
    """opening_scan_s=2.0, 60fps, target=3fps → range(0, 120, 20) = 6 frames."""
    sampler = AdaptivePresenceSampler(opening_scan_s=2.0, target_yolo_fps=3.0)
    sampler.last_source_fps = 60.0
    assert sampler._compute_opening_indices() == [0, 20, 40, 60, 80, 100]


def test_opening_scan_indices_one_second():
    """opening_scan_s=1.0, 60fps, target=3fps → range(0, 60, 20) = [0, 20, 40]."""
    sampler = AdaptivePresenceSampler(opening_scan_s=1.0, target_yolo_fps=3.0)
    sampler.last_source_fps = 60.0
    assert sampler._compute_opening_indices() == [0, 20, 40]


def test_opening_scan_zero_returns_empty():
    """opening_scan_s=0 disables the feature."""
    sampler = AdaptivePresenceSampler(opening_scan_s=0.0, target_yolo_fps=3.0)
    sampler.last_source_fps = 60.0
    assert sampler._compute_opening_indices() == []


def test_opening_scan_deduped_with_presence_windows():
    """Opening frame 0 merged with existing indices produces no duplicates."""
    sampler = AdaptivePresenceSampler(opening_scan_s=1.0, target_yolo_fps=3.0)
    sampler.last_source_fps = 60.0
    existing = [0, 15, 30, 45, 60, 75]
    opening = sampler._compute_opening_indices()  # [0, 20, 40]
    merged = sorted(set(existing) | set(opening))
    assert merged.count(0) == 1
    assert 20 in merged
    assert 40 in merged
```

- [ ] **Step 2: Verify they fail**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py::test_opening_scan_indices_two_seconds -v 2>&1 | tail -5
```

Expected: FAIL — `AttributeError: 'AdaptivePresenceSampler' object has no attribute '_compute_opening_indices'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_sampler_fast_scan.py
git commit -m "test(sampler): failing tests for opening scan feature"
```

---

### Task 3: Implement Feature A — opening scan in `AdaptivePresenceSampler`

**Files:**
- Modify: `src/card_capture/sampler/__init__.py`

- [ ] **Step 1: Add `opening_scan_s` parameter to `__init__`**

In `AdaptivePresenceSampler.__init__` (around line 336), after the `target_yolo_fps: float = 3.0` parameter:

```python
        target_yolo_fps: float = 3.0,
        opening_scan_s: float = 2.0,
```

In the `__init__` body, after `self.target_yolo_fps = max(0.1, target_yolo_fps)`:

```python
        self.target_yolo_fps = max(0.1, target_yolo_fps)
        self.opening_scan_s = max(0.0, opening_scan_s)
```

- [ ] **Step 2: Add `_compute_opening_indices` method**

Add this method to `AdaptivePresenceSampler`, directly after `_score_sharpness_in_window` ends (around line 700):

```python
    def _compute_opening_indices(self) -> list[int]:
        """Return source frame indices for the unconditional opening scan window.

        Cards resting on a stand at the start of the video may not register
        as 'card present' in the fast scan because they are static and score
        below the presence threshold — the scanner treats them as stable
        background. We unconditionally include sparse frames from the opening
        window so that a card placed before filming started is never silently
        skipped.

        This is distinct from the presence-gated windows: these frames are
        always sent to YOLO regardless of presence score. If no card is
        present, YOLO returns no detections and the cost is negligible
        (triage + inference on a handful of frames).

        Returns:
            Frame indices [0, stride, 2*stride, ...] up to opening_scan_s seconds
            of source video, at target_yolo_fps density. Empty list if
            opening_scan_s <= 0.
        """
        if self.opening_scan_s <= 0:
            return []
        source_fps = self.last_source_fps or 30.0
        opening_count = int(source_fps * self.opening_scan_s)
        stride = max(1, int(source_fps / self.target_yolo_fps))
        return list(range(0, opening_count, stride))
```

- [ ] **Step 3: Apply opening indices in `sample()`**

In `AdaptivePresenceSampler.sample()`, replace lines 866–869 (the existing block that builds `deduped_frame_indices` and calls `_decode_selected_frames`):

```python
        deduped_frame_indices = sorted(set(selected_frame_indices))

        # Merge opening-scan indices into the selected set.
        # See _compute_opening_indices for why this is necessary: cards resting
        # on a stand before presentation begins are invisible to the presence
        # gate and would be silently dropped without this forced inclusion.
        opening_indices = self._compute_opening_indices()
        if opening_indices:
            deduped_frame_indices = sorted(set(deduped_frame_indices) | set(opening_indices))

        self.last_scan_frame_count = len(self._scan_frames)
        self.last_selected_frame_count = len(deduped_frame_indices)
        yield from self._decode_selected_frames(resolved_video_path, deduped_frame_indices)
```

- [ ] **Step 4: Run the new tests**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py::test_opening_scan_indices_two_seconds tests/test_sampler_fast_scan.py::test_opening_scan_indices_one_second tests/test_sampler_fast_scan.py::test_opening_scan_zero_returns_empty tests/test_sampler_fast_scan.py::test_opening_scan_deduped_with_presence_windows -v 2>&1 | tail -10
```

Expected: all 4 PASS.

- [ ] **Step 5: Run full sampler test file**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/sampler/__init__.py
git commit -m "feat(sampler): always scan opening N seconds — catch cards resting on stand before presentation"
```

---

### Task 4: Write failing tests for `_laplacian_select_frames`

**Files:**
- Create: `tests/pipeline/test_laplacian_select.py`

- [ ] **Step 1: Create test file**

```python
"""Tests for _laplacian_select_frames — in-track Laplacian quality scan."""
import numpy as np
import pytest


def _make_video(tmp_path, frames: list[np.ndarray], fps: int = 30) -> "Path":
    """Write a synthetic video and return its path."""
    import cv2
    path = tmp_path / "test.mp4"
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in frames:
        out.write(f if f.ndim == 3 else np.stack([f] * 3, axis=-1))
    out.release()
    return path


def _blurry() -> np.ndarray:
    """Uniform gray — zero Laplacian variance."""
    return np.full((64, 64, 3), 128, dtype=np.uint8)


def _sharp() -> np.ndarray:
    """Checkerboard pattern — high Laplacian variance."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[::4, :] = 255
    img[:, ::4] = 255
    return img


CORNERS = [[0, 0], [63, 0], [63, 63], [0, 63]]


def test_selects_sharpest_frame(tmp_path):
    """Given blurry-sharp-blurry, frame 1 (sharp) is selected."""
    from card_capture.pipeline_utils import _laplacian_select_frames
    vpath = _make_video(tmp_path, [_blurry(), _sharp(), _blurry()])
    track_ranges = [{
        "instance_id": "abc",
        "detections": [(0, CORNERS), (1, CORNERS), (2, CORNERS)],
    }]
    result = _laplacian_select_frames(vpath, track_ranges, scan_stride=1, top_k=1, max_corner_gap=5)
    assert "abc" in result
    assert len(result["abc"]) == 1
    assert result["abc"][0][0] == 1  # frame index 1 = sharp


def test_empty_track_ranges(tmp_path):
    """Empty input returns empty dict without errors."""
    from card_capture.pipeline_utils import _laplacian_select_frames
    result = _laplacian_select_frames(tmp_path / "nonexistent.mp4", [], scan_stride=4, top_k=1, max_corner_gap=15)
    assert result == {}


def test_fallback_when_corner_gap_exceeded(tmp_path):
    """Sharpest frame far from any detection falls back to nearest detection."""
    from card_capture.pipeline_utils import _laplacian_select_frames
    # 6 blurry + 1 sharp at frame 5; detection only at frame 0 (gap=5 == max_corner_gap=4)
    frames = [_blurry()] * 5 + [_sharp()]
    vpath = _make_video(tmp_path, frames)
    track_ranges = [{
        "instance_id": "xyz",
        "detections": [(0, CORNERS)],  # only detection at frame 0
    }]
    result = _laplacian_select_frames(vpath, track_ranges, scan_stride=1, top_k=1, max_corner_gap=4)
    # Frame 5 is sharpest but gap=5 > max=4, so falls back to nearest detection = frame 0
    assert result["xyz"][0][0] == 0


def test_borrows_corners_from_nearest_detection(tmp_path):
    """Non-YOLO sharp frame borrows corners from nearest detection."""
    from card_capture.pipeline_utils import _laplacian_select_frames
    alt_corners = [[1, 1], [62, 1], [62, 62], [1, 62]]
    frames = [_blurry(), _sharp(), _blurry()]
    vpath = _make_video(tmp_path, frames)
    # Detections at 0 and 2; frame 1 (sharpest) is not a detection
    track_ranges = [{
        "instance_id": "abc",
        "detections": [(0, CORNERS), (2, alt_corners)],
    }]
    result = _laplacian_select_frames(vpath, track_ranges, scan_stride=1, top_k=1, max_corner_gap=5)
    fi, corners = result["abc"][0]
    assert fi == 1
    # Nearest detection to frame 1 is frame 0 or frame 2 (equal distance); corners from either
    assert corners == CORNERS or corners == alt_corners


def test_top_k_returns_multiple_frames(tmp_path):
    """top_k=2 returns two frames ordered sharpest first."""
    from card_capture.pipeline_utils import _laplacian_select_frames
    frames = [_blurry(), _sharp(), _blurry(), _sharp()]
    vpath = _make_video(tmp_path, frames)
    track_ranges = [{
        "instance_id": "abc",
        "detections": [(i, CORNERS) for i in range(4)],
    }]
    result = _laplacian_select_frames(vpath, track_ranges, scan_stride=1, top_k=2, max_corner_gap=5)
    assert len(result["abc"]) == 2
    sharp_frames = {result["abc"][0][0], result["abc"][1][0]}
    assert 1 in sharp_frames and 3 in sharp_frames  # the two sharp frames
```

- [ ] **Step 2: Verify they fail**

```bash
python3 -m pytest tests/pipeline/test_laplacian_select.py -v 2>&1 | tail -10
```

Expected: FAIL — `ImportError: cannot import name '_laplacian_select_frames'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/pipeline/test_laplacian_select.py
git commit -m "test(pipeline_utils): failing tests for _laplacian_select_frames"
```

---

### Task 5: Implement `_laplacian_select_frames` in `pipeline_utils.py`

**Files:**
- Modify: `src/card_capture/pipeline_utils.py`

- [ ] **Step 1: Add the function**

Append to `src/card_capture/pipeline_utils.py`:

```python
def _laplacian_select_frames(
    video_path: "Path",
    track_ranges: list,
    scan_stride: int = 4,
    top_k: int = 1,
    max_corner_gap: int = 15,
) -> dict:
    """Single-pass Laplacian sharpness scan across all confirmed track time ranges.

    After tracking confirms which time windows contain cards, this function
    finds the sharpest source frames within each track's window. It decouples
    detection coverage (temporal stride, 3fps) from output quality (dense
    sharpness scan within confirmed holds, ~15fps for scan_stride=4 at 60fps).

    Uses ONE forward video pass covering all track ranges — no repeated seeks.
    Frames are downscaled to 640px wide before Laplacian computation to keep
    per-frame cost under 2ms.

    Args:
        video_path: Absolute path to the source video file.
        track_ranges: List of track dicts, each with:
            - "instance_id": str
            - "detections": list of (frame_index: int, corners: list) tuples
              sorted by frame_index. These are the YOLO-detected frames.
        scan_stride: Decode every Nth source frame within each range.
            4 → ~15fps for a 60fps source, covering a 2-second hold in ~30 frames.
        top_k: Number of sharpest frames to return per track (≥ 1).
        max_corner_gap: Max distance in source frames between a selected frame
            and its nearest YOLO detection when borrowing corners. If the gap
            exceeds this, fall back to the nearest detection frame itself (safe
            — always has corners from YOLO). 15 ≈ 0.25s at 60fps.

    Returns:
        Dict mapping instance_id → list of (frame_index, corners) tuples,
        length ≤ top_k, ordered sharpest-first. Corners are either the frame's
        own (if it was a YOLO detection) or borrowed from the nearest detection.
        Returns empty dict if track_ranges is empty or video cannot be opened.
    """
    if not track_ranges:
        return {}

    import bisect
    import cv2
    import numpy as np

    # Build per-track metadata and collect all frame indices to scan
    track_info: dict = {}
    all_scan_frames: set = set()

    for t in track_ranges:
        iid = t["instance_id"]
        dets = sorted(t.get("detections", []), key=lambda x: x[0])
        if not dets:
            continue
        first_frame, last_frame = dets[0][0], dets[-1][0]
        det_map = {fi: corners for fi, corners in dets}
        det_sorted = [fi for fi, _ in dets]
        scan_frames = set(range(first_frame, last_frame + 1, scan_stride))
        track_info[iid] = {
            "first": first_frame,
            "last": last_frame,
            "det_map": det_map,
            "det_sorted": det_sorted,
            "scan_frames": scan_frames,
            "scores": {},   # frame_index → laplacian variance
        }
        all_scan_frames |= scan_frames

    if not all_scan_frames:
        return {}

    # Single forward video pass — compute Laplacian for every scan frame
    max_scan_frame = max(all_scan_frames)
    try:
        capture = _open_capture(video_path)
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

    # Select top_k sharpest frames per track; map each to corners
    result: dict = {}
    for iid, ti in track_info.items():
        if not ti["scores"]:
            result[iid] = []
            continue

        ranked = sorted(ti["scores"].items(), key=lambda x: -x[1])
        selected = []
        for frame_idx, _ in ranked:
            if len(selected) >= top_k:
                break
            det_sorted = ti["det_sorted"]
            if not det_sorted:
                continue

            # Find nearest YOLO detection to borrow corners from
            pos = bisect.bisect_left(det_sorted, frame_idx)
            candidates = []
            if pos < len(det_sorted):
                candidates.append(det_sorted[pos])
            if pos > 0:
                candidates.append(det_sorted[pos - 1])
            nearest = min(candidates, key=lambda f: abs(f - frame_idx))

            if abs(nearest - frame_idx) <= max_corner_gap:
                # Close enough — use this frame with borrowed corners
                selected.append((frame_idx, ti["det_map"][nearest]))
            else:
                # Too far — fall back to the nearest detection frame itself
                selected.append((nearest, ti["det_map"][nearest]))

        result[iid] = selected

    return result
```

- [ ] **Step 2: Run the 5 new tests**

```bash
python3 -m pytest tests/pipeline/test_laplacian_select.py -v 2>&1 | tail -12
```

Expected: all 5 PASS.

- [ ] **Step 3: Run broader suite**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py tests/pipeline/ -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/card_capture/pipeline_utils.py
git commit -m "feat(pipeline_utils): _laplacian_select_frames — single-pass in-track sharpness scan"
```

---

### Task 6: Integrate Laplacian scan into the refine step

**Files:**
- Modify: `pipeline/steps/refine.py`

The refine step currently:
1. Collects `canonical_indices` from track candidates (line 79–83)
2. Decodes those frames from video (line 86–100)
3. For each track, sorts candidates by `score_total` and takes top 8 (line 142)

We insert the Laplacian scan between steps 1 and 2, then adjust the top-8 selection in step 3 to prefer Laplacian-ranked frames.

- [ ] **Step 1: Import and run Laplacian scan before the decode loop**

Replace lines 78–100 in `pipeline/steps/refine.py` with:

```python
    # Determine which high-res frames to decode
    canonical_indices: set = set()
    for track_dict in tracks_data:
        for c in track_dict["candidates"]:
            if c["frame_index"] is not None:
                canonical_indices.add(int(c["frame_index"]))

    # --- In-track Laplacian quality scan ---
    # Temporal stride selected frames at ~3fps per track. Within each confirmed
    # track's time window we scan densely (laplacian_scan_stride source frames)
    # to find sharper frames that the sparse YOLO pass may have skipped.
    # This runs as ONE forward video pass across all tracks — no repeated seeks.
    # Non-YOLO frames use corners borrowed from the nearest YOLO detection
    # (corners are stable over short holds; max_corner_gap_frames limits drift).
    from card_capture.pipeline_utils import _laplacian_select_frames
    _lap_top_k = max(1, ctx.fusion_target_frames)
    _lap_ranges = []
    for _td in tracks_data:
        _dets = [
            (int(c["frame_index"]), c["corners"])
            for c in _td["candidates"]
            if c["frame_index"] is not None
        ]
        if _dets:
            _lap_ranges.append({"instance_id": _td["instance_id"], "detections": _dets})

    _lap_results: dict = {}
    try:
        _lap_results = _laplacian_select_frames(
            video_path,
            _lap_ranges,
            scan_stride=ctx.laplacian_scan_stride,
            top_k=_lap_top_k,
            max_corner_gap=ctx.max_corner_gap_frames,
        )
    except Exception as _e:
        print(f"[Refine] Laplacian scan failed, falling back to temporal-stride frames: {_e}")

    # Add any non-YOLO Laplacian-selected frames to the canonical decode set
    for _sel_list in _lap_results.values():
        for _fi, _ in _sel_list:
            canonical_indices.add(int(_fi))
    # ----------------------------------------

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

- [ ] **Step 2: Prefer Laplacian-selected frames in the per-track loop**

In the per-track loop, after line 142 (`scored_candidates = sorted(candidates_data, key=lambda c: c["score_total"], reverse=True)[:8]`), add:

```python
        # Sort by score and take top 8 for canonical selection
        scored_candidates = sorted(candidates_data, key=lambda c: c["score_total"], reverse=True)[:8]

        # Reorder: put Laplacian-selected frames first. This overrides the
        # quality-score ranking so the sharpest frame(s) from the dense scan
        # become the canonical output. Frames not in the YOLO detection set
        # (picked from between detections) are inserted as synthetic candidates
        # with borrowed corners; the warp logic treats them identically.
        _lap_frames = _lap_results.get(instance_id, [])
        if _lap_frames:
            _existing_fi = {c.get("frame_index") for c in candidates_data}
            _source_fps = track_out.sampler_telemetry.get("last_source_fps", 30.0) or 30.0
            _lap_leading = []
            _lap_fi_set = set()
            for _fi, _corners in _lap_frames:
                _lap_fi_set.add(_fi)
                if _fi in _existing_fi:
                    # Already a YOLO detection — pull it to front with its own data
                    _match = next((c for c in candidates_data if c.get("frame_index") == _fi), None)
                    if _match:
                        _lap_leading.append(_match)
                else:
                    # Non-YOLO frame: synthesize a minimal candidate using
                    # borrowed corners. score_total is a placeholder — ordering
                    # here is by Laplacian sharpness, not by this score.
                    _ref = candidates_data[0] if candidates_data else {}
                    _lap_leading.append({
                        "frame_index": _fi,
                        "corners": _corners,
                        "score_total": float(np.median([c["score_total"] for c in candidates_data])) if candidates_data else 0.7,
                        "detection_id": -1,
                        "confidence": float(np.median([c["confidence"] for c in candidates_data])) if candidates_data else 0.7,
                        "width": _ref.get("width", 3840),
                        "height": _ref.get("height", 2160),
                        "timestamp_ms": int(_fi * 1000 / _source_fps),
                    })
            # Lap frames first, then remaining scored candidates not already included
            _remaining = [c for c in scored_candidates if c.get("frame_index") not in _lap_fi_set]
            scored_candidates = (_lap_leading + _remaining)[:8]
```

- [ ] **Step 3: Run existing tests**

```bash
python3 -m pytest tests/test_sampler_fast_scan.py tests/pipeline/test_laplacian_select.py tests/pipeline/test_score_novelty_gate.py -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add pipeline/steps/refine.py
git commit -m "$(cat <<'EOF'
feat(refine): in-track Laplacian quality scan replaces temporal-stride ranking

After tracking confirms each card's time window, a single forward video
pass scans at laplacian_scan_stride density to find the sharpest frames.
These are preferred over the temporal-stride selections for the Kornia
warp, decoupling detection coverage (3fps) from output quality (15fps scan).
Non-YOLO frames use borrowed corners from the nearest detection.

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```
