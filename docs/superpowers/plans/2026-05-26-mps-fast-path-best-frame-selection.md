# MPS Fast Path — Best-Frame Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On Apple Silicon (MPS / `docaligner`), keep dense stride sampling but do the expensive high-res Kornia warp on at most 5 frames per tracked card — the flattest + clearest — and make the route MPS-or-fail (no silent CPU fallback).

**Architecture:** Reuse existing machinery rather than build new. `_run_fused_inference` already skips the eager warp when no `crop_cache` is passed; the non-fused `refine.run` already does select→decode→warp→score→pick-best. So: (1) add two cheap, pure scorers (flatness from corner geometry, clarity from a GPU ROI Laplacian); (2) have the MPS detect loop compute and store them per detection without warping; (3) route MPS through the standard non-fused pipeline (CUDA keeps `fused_refine` unchanged); (4) drive refine's candidate selection from the cheap scores, capped at 5, skipping the now-redundant Laplacian rescan; (5) enforce MPS-or-fail.

**Tech Stack:** Python, PyTorch (MPS), Ultralytics YOLOv8-OBB, Kornia, OpenCV, pytest.

> **Realization note (read before starting):** The approved spec (`docs/superpowers/specs/2026-05-26-mps-fast-path-best-frame-selection-design.md`) described "the fused path." During planning we found the fused single-step is CUDA-specific and that the standard non-fused path already implements select-then-warp. This plan therefore routes MPS through the non-fused path and leaves CUDA's `fused_refine` untouched. Behavior matches the spec exactly; only the integration point differs.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/card_capture/frame_quality.py` | **Create** | Pure scorers: `flatness_score(corners)` and `clarity_score_gpu(frame_hwc, bbox, device)`. |
| `tests/test_frame_quality.py` | **Create** | Unit tests for both scorers (CPU device — no MPS needed in CI). |
| `src/card_capture/gpu_utils.py` | Modify | Add `require_device(requested)` that raises on silent CPU downgrade. |
| `tests/test_require_device.py` | **Create** | Verify `require_device` raises when the requested accelerator is unavailable. |
| `pipeline/steps/detect.py` | Modify | In `_run_fused_inference`: compute+store flatness/clarity per detection (no warp when `crop_cache is None`); MPS `batch_size`; drop per-5-batch `mps.empty_cache()`; MPS-or-fail guard. |
| `pipeline/card_capture_flow.py` | Modify | Route `mps`/`docaligner` through the non-fused path; keep `cuda` on `fused_refine`. |
| `pipeline/steps/refine.py` | Modify | When detections carry cheap scores, select top-5/track by flat+clear and skip the Laplacian rescan. |
| `pipeline/steps/detect_fused_helper.py` | **Delete** | Abandoned duplicate. |
| `debug_pipeline_thumb.jpg`, `diagnostic_worker_thumb.jpg`, `test_thumb_gpu.jpg` | **Delete** | Stray debug artifacts. |

---

## Task 1: Flatness scorer (corner geometry)

**Files:**
- Create: `src/card_capture/frame_quality.py`
- Test: `tests/test_frame_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frame_quality.py
import math
import numpy as np
from card_capture.frame_quality import flatness_score


def test_perfect_5x7_rectangle_scores_near_one():
    # Axis-aligned rectangle with the card's 5:7 short:long ratio (500x700).
    corners = [(0.0, 0.0), (500.0, 0.0), (500.0, 700.0), (0.0, 700.0)]
    assert flatness_score(corners) > 0.95


def test_keystoned_quad_scores_lower_than_rectangle():
    rect = [(0.0, 0.0), (500.0, 0.0), (500.0, 700.0), (0.0, 700.0)]
    # Top edge much narrower than bottom — strong perspective skew.
    keystone = [(150.0, 0.0), (350.0, 0.0), (500.0, 700.0), (0.0, 700.0)]
    assert flatness_score(keystone) < flatness_score(rect)
    assert flatness_score(keystone) < 0.8


def test_corner_order_independent():
    rect = [(0.0, 0.0), (500.0, 0.0), (500.0, 700.0), (0.0, 700.0)]
    shuffled = [rect[2], rect[0], rect[3], rect[1]]
    assert abs(flatness_score(rect) - flatness_score(shuffled)) < 1e-6


def test_returns_zero_for_degenerate_input():
    assert flatness_score([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]) == 0.0
    assert flatness_score([(0.0, 0.0)]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_frame_quality.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'card_capture.frame_quality'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/card_capture/frame_quality.py
"""Cheap, warp-free quality scorers used to shortlist frames before the
expensive Kornia warp. Both are pure and independently testable."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Point = Tuple[float, float]

# Card canonical short:long ratio (750x1050).
_TARGET_RATIO = 750.0 / 1050.0


def _order_corners(corners: Sequence[Point]) -> List[Point]:
    """Order 4 points counter-clockwise around their centroid (order-invariant)."""
    cx = sum(p[0] for p in corners) / 4.0
    cy = sum(p[1] for p in corners) / 4.0
    return sorted(corners, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(prev: Point, vert: Point, nxt: Point) -> float:
    """Interior angle at `vert` in degrees."""
    v1 = (prev[0] - vert[0], prev[1] - vert[1])
    v2 = (nxt[0] - vert[0], nxt[1] - vert[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def flatness_score(corners: Sequence[Point]) -> float:
    """Return how flat/fronto-parallel a detected card quad is, in [0, 1].

    Combines three deterministic terms:
      - aspect: closeness of the quad's short:long ratio to the 5:7 card ratio
      - side balance: opposite sides near-equal (no perspective foreshortening)
      - angles: interior angles near 90 degrees (rectangular, not sheared)
    Returns 0.0 for degenerate input (fewer than 4 points or zero area).
    """
    if corners is None or len(corners) != 4:
        return 0.0
    pts = _order_corners([(float(x), float(y)) for x, y in corners])
    sides = [_dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    if min(sides) < 1e-3:
        return 0.0

    s0, s1, s2, s3 = sides  # consecutive edges; (s0,s2) opposite, (s1,s3) opposite
    w = (s0 + s2) / 2.0
    h = (s1 + s3) / 2.0
    short, long_ = (w, h) if w <= h else (h, w)
    quad_ratio = short / long_  # in (0, 1]
    aspect_term = max(0.0, 1.0 - abs(quad_ratio - _TARGET_RATIO) / _TARGET_RATIO)

    bal0 = 1.0 - abs(s0 - s2) / (s0 + s2)
    bal1 = 1.0 - abs(s1 - s3) / (s1 + s3)
    side_term = (bal0 + bal1) / 2.0

    angles = [_angle(pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]) for i in range(4)]
    angle_term = max(0.0, 1.0 - sum(abs(a - 90.0) for a in angles) / 4.0 / 90.0)

    score = 0.4 * aspect_term + 0.3 * side_term + 0.3 * angle_term
    return max(0.0, min(1.0, score))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_frame_quality.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/frame_quality.py tests/test_frame_quality.py
git commit -m "feat(mps): add corner-geometry flatness scorer"
```

---

## Task 2: Clarity scorer (GPU ROI Laplacian variance)

**Files:**
- Modify: `src/card_capture/frame_quality.py`
- Test: `tests/test_frame_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_frame_quality.py
import cv2
from card_capture.frame_quality import clarity_score_gpu


def _checkerboard(h=200, w=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[::8, :, :] = 255
    img[:, ::8, :] = 255
    return img


def test_sharp_scores_higher_than_blurred_cpu_device():
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=4)
    bbox = (0, 0, 200, 200)  # x0, y0, x1, y1
    s_sharp = clarity_score_gpu(sharp, bbox, device="cpu")
    s_blur = clarity_score_gpu(blurred, bbox, device="cpu")
    assert s_sharp > s_blur
    assert s_blur >= 0.0


def test_clarity_uses_only_the_roi():
    # Sharp texture only inside the ROI; flat elsewhere → ROI score stays high.
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = _checkerboard(100, 100)
    full = clarity_score_gpu(img, (0, 0, 200, 200), device="cpu")
    roi = clarity_score_gpu(img, (50, 50, 150, 150), device="cpu")
    assert roi > full
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_frame_quality.py -q`
Expected: FAIL — `ImportError: cannot import name 'clarity_score_gpu'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to src/card_capture/frame_quality.py
import numpy as np


def clarity_score_gpu(frame_hwc: "np.ndarray", bbox: Tuple[int, int, int, int],
                      device: str = "auto") -> float:
    """Laplacian-variance sharpness of the card's bbox ROI, computed on `device`.

    `frame_hwc` is an HxWxC (BGR) uint8 array. `bbox` is (x0, y0, x1, y1) in
    pixel coords. Returns a raw variance >= 0 (higher = sharper); callers
    normalize per-track. Used as the cheap clarity signal before any warp.
    """
    import torch
    import torch.nn.functional as F
    import cv2

    x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    h, w = frame_hwc.shape[:2]
    x0 = max(0, min(x0, w - 1)); x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(y0 + 1, min(y1, h))
    roi = frame_hwc[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    dev = torch.device(device if device != "auto" else
                       ("mps" if torch.backends.mps.is_available()
                        else ("cuda" if torch.cuda.is_available() else "cpu")))
    t = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                          dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(t, kernel, padding=1)
    return float(torch.var(lap).item() * (255.0 ** 2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_frame_quality.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/frame_quality.py tests/test_frame_quality.py
git commit -m "feat(mps): add GPU ROI clarity (Laplacian variance) scorer"
```

---

## Task 3: MPS-or-fail device guard

**Files:**
- Modify: `src/card_capture/gpu_utils.py`
- Test: `tests/test_require_device.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_require_device.py
import pytest
import torch
from card_capture import gpu_utils


def test_require_device_mps_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS .*not available"):
        gpu_utils.require_device("mps")


def test_require_device_returns_requested_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert gpu_utils.require_device("mps").type == "mps"


def test_require_device_cpu_is_explicit_and_allowed():
    # Explicit CPU is intentional (tests/CI), not a silent downgrade.
    assert gpu_utils.require_device("cpu").type == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_require_device.py -q`
Expected: FAIL — `AttributeError: module 'card_capture.gpu_utils' has no attribute 'require_device'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to src/card_capture/gpu_utils.py
def require_device(requested: str) -> torch.device:
    """Resolve `requested` to a torch.device, raising rather than silently
    downgrading to CPU. `requested` of "cpu" is honored (explicit). "auto"
    requires a GPU (cuda or mps) and raises if none is present.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available "
                               "(this route is MPS-or-fail; no CPU fallback).")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        return torch.device("cuda")
    # auto: require some accelerator
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("No GPU (cuda/mps) available and this route forbids "
                       "silent CPU fallback. Set device='cpu' explicitly to override.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_require_device.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/gpu_utils.py tests/test_require_device.py
git commit -m "feat(mps): add require_device guard (no silent CPU fallback)"
```

---

## Task 4: Detect loop — store cheap scores, no warp on MPS, MPS-native tuning

**Files:**
- Modify: `pipeline/steps/detect.py` (function `_run_fused_inference`, lines ~207-430; helper `_build_sampler_detector` lines ~175-203)
- Test: `tests/pipeline/test_detect_cheap_scores.py`

This task changes **only** the `crop_cache is None` branch (the MPS/no-warp path). The `crop_cache is not None` branch (CUDA fused) is left intact so CUDA behavior does not change.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_detect_cheap_scores.py
"""The MPS detect path must attach flatness+clarity to each detection row
and must NOT eager-warp when no crop_cache is supplied."""
import numpy as np
from pipeline.steps import detect as detect_mod


def test_detection_rows_carry_flatness_and_clarity(monkeypatch):
    # Drive _annotate_cheap_scores directly: it is the unit under test.
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    frame[100:800, 100:600] = 255  # a bright rectangle
    corners = [(100.0, 100.0), (600.0, 100.0), (600.0, 800.0), (100.0, 800.0)]
    row = {"detection_id": 0, "corners": corners, "triage_metrics": {}}
    detect_mod._annotate_cheap_scores(row, frame, device="cpu")
    assert "flatness" in row["triage_metrics"]
    assert "clarity" in row["triage_metrics"]
    assert row["triage_metrics"]["flatness"] > 0.9
    assert row["triage_metrics"]["clarity"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/pipeline/test_detect_cheap_scores.py -q`
Expected: FAIL — `AttributeError: module 'pipeline.steps.detect' has no attribute '_annotate_cheap_scores'`

- [ ] **Step 3: Add the helper and wire it in**

Add this module-level helper to `pipeline/steps/detect.py` (top level, after imports):

```python
def _annotate_cheap_scores(row: dict, frame_bgr, device: str = "auto") -> None:
    """Attach warp-free flatness + clarity to a detection row's triage_metrics.

    `frame_bgr` is the full-res HxWxC BGR frame (numpy). Corners are in
    full-res pixel coords. Computed before any Kornia warp so it is cheap.
    """
    from card_capture.frame_quality import flatness_score, clarity_score_gpu
    corners = row.get("corners") or []
    flatness = flatness_score(corners)
    clarity = 0.0
    if corners:
        xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        clarity = clarity_score_gpu(frame_bgr, bbox, device=device)
    tm = row.setdefault("triage_metrics", {})
    tm["flatness"] = round(float(flatness), 6)
    tm["clarity"] = round(float(clarity), 6)
```

In `_run_fused_inference`, inside the `elif packets_out:` branch (the no-crop-cache path, ~line 384), after the `detection_rows.append({...})` for each `pkt`, annotate using the full-res frame. The full-res BGR is already on the GPU as `full_res_bgr_t`; read the per-frame slice back to numpy once for the ROI score:

```python
        elif packets_out:
            import numpy as np
            pos_by_frame = {f.frame_index: i for i, f in enumerate(frames)}
            for pkt in packets_out:
                cd = pkt.corner_detection
                row = {
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
                _bi = pos_by_frame.get(pkt.frame_index)
                if _bi is not None:
                    frame_np = full_res_bgr_t[_bi].to("cpu").numpy()
                    _annotate_cheap_scores(row, frame_np, device=resolved_device)
                detection_rows.append(row)
                det_id += 1
```

Change MPS batch size (line ~258) and drop the per-5-batch empty_cache (lines ~402-404):

```python
    # Larger batch is fine on unified memory; 4 was over-conservative.
    batch_size = 12 if resolved_device == "mps" else 8
```

Delete these three lines (the periodic stall):

```python
        if yolo_batches % 5 == 0:
            if resolved_device == "mps":
                torch.mps.empty_cache()
```

Add the MPS-or-fail guard: replace `resolved_device = probe_torch_device_status(ctx.kornia_device).resolved` (line ~240) with a strict resolve when the requested device is mps:

```python
    from card_capture.detectors import probe_torch_device_status
    from card_capture import gpu_utils
    if ctx.kornia_device == "mps":
        resolved_device = gpu_utils.require_device("mps").type  # raises, no CPU fallback
    else:
        resolved_device = probe_torch_device_status(ctx.kornia_device).resolved
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/pipeline/test_detect_cheap_scores.py tests/pipeline/test_detect_cuda.py tests/pipeline/test_detect_crop_cache.py tests/pipeline/test_detect_prefetch.py -q`
Expected: PASS (cheap-scores test passes; existing cuda/crop-cache/prefetch tests still pass — CUDA branch unchanged)

- [ ] **Step 5: Commit**

```bash
git add pipeline/steps/detect.py tests/pipeline/test_detect_cheap_scores.py
git commit -m "feat(mps): cheap flatness/clarity in detect, no eager warp, MPS-native tuning"
```

---

## Task 5: Route MPS through the non-fused pipeline

**Files:**
- Modify: `pipeline/card_capture_flow.py` (the `detect` step, lines ~79-99; the `refine` step branch lines ~124-146)

CUDA keeps the fused single-step (`fused_refine`); MPS/`docaligner` use the normal `detect → novelty → track → refine` flow so refine's select-then-warp path runs.

- [ ] **Step 1: Edit the detect step branch**

Change the condition at `pipeline/card_capture_flow.py:83` from:

```python
        if ctx.detector in ("cuda", "mps", "docaligner"):
            # Fused path: decode + warp + novelty + track + refine in one step
            ...
            self.refine_out = fused_refine.run(ctx)
            self._fused = True
```

to:

```python
        if ctx.detector == "cuda":
            # CUDA fused path: decode + warp + novelty + track + refine in one step
            ...
            self.refine_out = fused_refine.run(ctx)
            self._fused = True
```

The existing `else:` branch (which calls `detect.run(ctx)` and sets `self._fused = False`) now also handles `mps`/`docaligner`. Leave it unchanged.

- [ ] **Step 2: Verify the flow imports/branches still parse**

Run: `python3 -c "import pipeline.card_capture_flow"`
Expected: no error.

- [ ] **Step 3: Run the existing pipeline tests**

Run: `python3 -m pytest tests/pipeline/ -q --ignore=tests/pipeline/test_path_equivalence.py`
Expected: PASS (no regressions; pre-existing known failures excluded per CLAUDE.md).

- [ ] **Step 4: Commit**

```bash
git add pipeline/card_capture_flow.py
git commit -m "feat(mps): route mps/docaligner through non-fused select-then-warp path"
```

---

## Task 6: Refine — select top-5 by flat+clear, skip redundant Laplacian rescan

**Files:**
- Modify: `pipeline/steps/refine.py` (function `run`, candidate selection at lines ~270-311; Laplacian-scan gate lines ~156-204)
- Test: `tests/pipeline/test_refine_cheap_selection.py`

When detection rows carry our cheap scores (`triage_metrics` has `flatness`), refine selects the top-5 candidates per track by a combined flat+clear score and skips the Laplacian rescan (redundant at 15 fps). Otherwise behavior is unchanged (CUDA fused via `decoded_crops`, and legacy detectors).

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_refine_cheap_selection.py
"""When triage_metrics carry cheap scores, refine should rank candidates by
combined flat+clear and cap the shortlist at 5."""
from pipeline.steps.refine import _select_cheap_candidates


def _cand(did, flatness, clarity):
    return {"detection_id": did, "frame_index": did, "corners": [(0, 0)] * 4,
            "score_total": 0.5, "confidence": 0.9}


def test_selects_top5_by_combined_flat_and_clear():
    cands = [_cand(i, flatness=i / 10.0, clarity=float(i)) for i in range(10)]
    lookup = {i: {"triage_metrics": {"flatness": i / 10.0, "clarity": float(i)}}
              for i in range(10)}
    picked = _select_cheap_candidates(cands, lookup, top_k=5)
    ids = [c["detection_id"] for c in picked]
    assert ids == [9, 8, 7, 6, 5]  # highest flat+clear first


def test_returns_all_when_fewer_than_k():
    cands = [_cand(i, i / 10.0, float(i)) for i in range(3)]
    lookup = {i: {"triage_metrics": {"flatness": i / 10.0, "clarity": float(i)}}
              for i in range(3)}
    assert len(_select_cheap_candidates(cands, lookup, top_k=5)) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/pipeline/test_refine_cheap_selection.py -q`
Expected: FAIL — `ImportError: cannot import name '_select_cheap_candidates'`

- [ ] **Step 3: Add the selector helper and use it**

Add this module-level helper to `pipeline/steps/refine.py`:

```python
def _select_cheap_candidates(candidates_data, detection_lookup, top_k=5):
    """Rank candidates by combined flatness + (per-track normalized) clarity and
    return the top_k. Used only when detections carry cheap scores (MPS path)."""
    def _scores(c):
        tm = detection_lookup.get(c["detection_id"], {}).get("triage_metrics", {})
        return float(tm.get("flatness", 0.0)), float(tm.get("clarity", 0.0))
    clarities = [_scores(c)[1] for c in candidates_data]
    max_clar = max(clarities) if clarities else 0.0
    def _combined(c):
        flat, clar = _scores(c)
        norm_clar = (clar / max_clar) if max_clar > 0 else 0.0
        return 0.5 * flat + 0.5 * norm_clar
    return sorted(candidates_data, key=_combined, reverse=True)[:top_k]


def _has_cheap_scores(candidates_data, detection_lookup) -> bool:
    for c in candidates_data:
        tm = detection_lookup.get(c["detection_id"], {}).get("triage_metrics", {})
        if "flatness" in tm:
            return True
    return False
```

Gate the Laplacian rescan so it is skipped for the cheap-score path. At line ~175 change:

```python
    if decoded_crops is None:
```

to compute a flag once (place this just before the `decoded_images` block, after `detection_lookup` is built ~line 140):

```python
    _cheap_mode = decoded_crops is None and any(
        "flatness" in detection_lookup.get(c["detection_id"], {}).get("triage_metrics", {})
        for td in tracks_data for c in td["candidates"]
    )
```

and change the two `if decoded_crops is None:` guards around the Laplacian scan (lines ~175 and ~185) to:

```python
    if decoded_crops is None and not _cheap_mode:
```

In the per-track loop, replace the top-8 selection (line ~275):

```python
        scored_candidates = sorted(candidates_data, key=lambda c: c["score_total"], reverse=True)[:8]
```

with:

```python
        if _cheap_mode:
            scored_candidates = _select_cheap_candidates(
                candidates_data, detection_lookup, top_k=5)
        else:
            scored_candidates = sorted(
                candidates_data, key=lambda c: c["score_total"], reverse=True)[:8]
```

The `_lap_frames` reordering block (lines ~282-311) is naturally inert in cheap mode because `_lap_results` is empty (scan skipped), so `_lap_frames` is `[]`.

For `_cheap_mode`, the decode set must still be populated (the scan no longer adds frames, but `canonical_indices` already holds every candidate frame from lines ~143-147, and we need the selected ones decoded). Ensure decode runs for cheap mode by changing the decode guard at line ~175 to decode whenever crops are absent:

```python
    decoded_images: Dict[int, np.ndarray] = {}
    _lap_scan_indices: set = set()
    _decode_frames_elapsed = 0.0
    if decoded_crops is None:
        if not _cheap_mode:
            _lap_scan_indices = _compute_laplacian_scan_indices(_lap_ranges, ctx.laplacian_scan_stride)
        _all_needed = canonical_indices | _lap_scan_indices
        _t_decode_start = time.time()
        if _all_needed:
            decoded_images = decode_frames_gpu(video_path, sorted(_all_needed))
        _decode_frames_elapsed = time.time() - _t_decode_start
```

> The single-best pick per card happens downstream exactly as today: `fusion_target_frames` defaults to `1`, so the existing canonical-selection/fusion path reduces the warped+scored shortlist to one image per track. No change needed there.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/pipeline/test_refine_cheap_selection.py tests/pipeline/test_fused_refine.py -q`
Expected: PASS (new selection test passes; fused refine test unchanged)

- [ ] **Step 5: Run the broader suite for regressions**

Run: `python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py`
Expected: PASS except the pre-existing known failures listed in CLAUDE.md.

- [ ] **Step 6: Commit**

```bash
git add pipeline/steps/refine.py tests/pipeline/test_refine_cheap_selection.py
git commit -m "feat(mps): refine selects top-5 by flat+clear, skips redundant laplacian rescan"
```

---

## Task 7: Harden the warp path — MPS-or-fail (no CPU normalizer fallback)

**Files:**
- Modify: `pipeline/steps/refine.py` (Kornia setup, lines ~206-220)
- Test: `tests/pipeline/test_refine_mps_or_fail.py`

Today `refine.run` resolves the device with `probe_torch_device_status` (silent CPU downgrade) and swallows a Kornia init failure into `kornia_normalizer = None`, falling back to the CPU `PrecisionNormalizer`. This task makes the warp MPS-or-fail: on a GPU device, init failure raises; explicit `cpu` still allows the None fallback.

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_refine_mps_or_fail.py
import pytest
from pipeline.steps.refine import _make_kornia_normalizer


class _Boom:
    def __init__(self, *a, **k):
        raise RuntimeError("kornia init failed")


def test_gpu_warp_failure_raises_not_cpu_fallback():
    # On a GPU device, a Kornia init failure must raise (MPS-or-fail),
    # never silently downgrade to the CPU normalizer.
    with pytest.raises(RuntimeError, match="kornia"):
        _make_kornia_normalizer(_Boom, use_kornia=True, device="mps",
                                width=750, height=1050)


def test_explicit_cpu_allows_none_fallback():
    # Explicit CPU is an intentional override, so None (CPU normalizer) is allowed.
    out = _make_kornia_normalizer(_Boom, use_kornia=True, device="cpu",
                                  width=750, height=1050)
    assert out is None


def test_use_kornia_disabled_returns_none():
    out = _make_kornia_normalizer(_Boom, use_kornia=False, device="mps",
                                  width=750, height=1050)
    assert out is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/pipeline/test_refine_mps_or_fail.py -q`
Expected: FAIL — `ImportError: cannot import name '_make_kornia_normalizer'`

- [ ] **Step 3: Add the helper and use it**

Add this module-level helper to `pipeline/steps/refine.py`:

```python
def _make_kornia_normalizer(kornia_cls, use_kornia, device, width, height):
    """Construct a KorniaNormalizer, or return None when warping is disabled or
    explicitly on CPU. On a GPU device (mps/cuda) an init failure RAISES — no
    silent CPU fallback (MPS-or-fail)."""
    if not use_kornia:
        return None
    try:
        return kornia_cls(width=width, height=height, device=device)
    except Exception:
        if device == "cpu":
            return None
        raise
```

Replace the device-resolve + Kornia setup block (lines ~206-220):

```python
    # Set up Kornia normalizer (MPS-or-fail: no silent CPU downgrade).
    from card_capture.detectors import probe_torch_device_status
    from card_capture import gpu_utils
    if ctx.kornia_device == "mps":
        resolved_device = gpu_utils.require_device("mps").type
    else:
        resolved_device = probe_torch_device_status(ctx.kornia_device).resolved

    normalizer = PrecisionNormalizer()
    kornia_normalizer = _make_kornia_normalizer(
        KorniaNormalizer, ctx.use_kornia, resolved_device,
        normalizer.width, normalizer.height,
    )
```

(The later `elif kornia_normalizer is not None ...` warp branch is unchanged; on the MPS route `kornia_normalizer` is now guaranteed non-None or we have already raised.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/pipeline/test_refine_mps_or_fail.py tests/pipeline/test_refine_cheap_selection.py tests/pipeline/test_fused_refine.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/steps/refine.py tests/pipeline/test_refine_mps_or_fail.py
git commit -m "feat(mps): warp path is MPS-or-fail, no silent CPU normalizer fallback"
```

---

## Task 8: Cleanup — delete abandoned scratch

**Files:**
- Delete: `pipeline/steps/detect_fused_helper.py`, `debug_pipeline_thumb.jpg`, `diagnostic_worker_thumb.jpg`, `test_thumb_gpu.jpg`

- [ ] **Step 1: Confirm the helper is unreferenced**

Run: `grep -rn "detect_fused_helper" --include="*.py" .`
Expected: no matches (only the file itself, which is being deleted).

- [ ] **Step 2: Delete the files**

```bash
git rm pipeline/steps/detect_fused_helper.py
rm -f debug_pipeline_thumb.jpg diagnostic_worker_thumb.jpg test_thumb_gpu.jpg
```

- [ ] **Step 3: Verify imports still resolve**

Run: `python3 -c "import pipeline.steps.detect, pipeline.steps.fused_refine, pipeline.card_capture_flow"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(mps): remove abandoned detect_fused_helper and debug images"
```

---

## Task 9: Integration verification on a real video

**Files:** none (manual run). Requires a test `.MOV` path — the user will provide it at execution time (referred to below as `$VIDEO`).

- [ ] **Step 1: Baseline (current branch tip before this work) — record timing + counts**

Run (substituting the provided path):
```bash
card-capture process "$VIDEO" --output-dir /tmp/mps_after --db /tmp/mps_after/cards.sqlite
```
Capture: wall-clock time, `run_telemetry.json` (`fused_inference_s`, refine timings, `device_resolved`), and number of `card_instances` rows in the DB.

- [ ] **Step 2: Confirm warp count dropped**

Inspect `run_telemetry.json` / `tracker_association_events.json` for the per-track warp/candidate counts. Expected: warps ≈ Σ min(5, candidates/track), far below total detections.

- [ ] **Step 3: Visual spot-check**

Open a sample of `/tmp/mps_after/crops/*.jpg`. Expected: cards are flat (fronto-parallel, full card framed) and sharp.

- [ ] **Step 4: Confirm MPS-or-fail**

Run with MPS force-disabled to prove no silent CPU path:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 card-capture process "$VIDEO" --output-dir /tmp/mps_off --db /tmp/mps_off/cards.sqlite --device mps
```
Expected: a clear `RuntimeError` about MPS unavailability if MPS is off — not a slow CPU run. (On a working Mac with MPS present, this simply runs on MPS.)

- [ ] **Step 5: Report results to the user** (timing before/after, warp-count reduction, card counts, visual quality) and decide whether to tune K, the flat/clear weights, or `batch_size`.

---

## Self-Review

**Spec coverage:**
- Keep StrideSampler dense coverage → unchanged (Task 4/5 leave sampler as-is).
- Cheap flatness + clarity inline, no warp → Tasks 1, 2, 4.
- Tracker groups cards; per-card top-5 → Task 6 (`_select_cheap_candidates`, top_k=5) over tracked candidates.
- Warp only selected, repick single best on warped pixels → Task 6 (decode+warp+score shortlist; `fusion_target_frames=1` picks one).
- Fusion bypassed → satisfied implicitly by `fusion_target_frames=1` (single canonical) on the non-fused path.
- MPS-or-fail, no CPU compute fallback → Task 3 (`require_device`) + Task 4 (detect device guard) + **Task 7 (warp path raises on GPU init failure, no CPU `PrecisionNormalizer` fallback)**. Full MPS-or-fail across detect and warp.
- Remove eager warp / empty_cache thrash / tiny batch → Task 4.
- Cleanup → Task 8.
- Testing (unit + integration) → Tasks 1-3, 6, 7 (unit), Task 9 (integration).

**Placeholder scan:** `$VIDEO` in Task 9 is a runtime input supplied by the user, not an unspecified design detail. No other placeholders.

**Type consistency:** `flatness_score(corners)->float`, `clarity_score_gpu(frame_hwc, bbox, device)->float`, `require_device(requested)->torch.device`, `_annotate_cheap_scores(row, frame_bgr, device)`, `_select_cheap_candidates(candidates_data, detection_lookup, top_k)`, `_has_cheap_scores(...)`, `_make_kornia_normalizer(kornia_cls, use_kornia, device, width, height)` are used consistently across tasks. `triage_metrics` keys `flatness`/`clarity` are written in Task 4 and read in Task 6.

**MPS-or-fail is now total:** device resolution raises rather than downgrading (Task 3/4), and the warp itself raises on GPU init failure instead of using the CPU `PrecisionNormalizer` (Task 7). The only CPU path remaining is the explicit `device="cpu"` override intended for tests/CI.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-mps-fast-path-best-frame-selection.md`.
