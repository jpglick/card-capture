# Fused Decode→Warp Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the redundant 16.2s second video decode in the `refine` stage by warping every detection to its 750×1050 canonical crop *during* the `detect` stage's decode pass (while the 4K frame is already in RAM), caching those crops in-process, and feeding them to a reused `refine` so it never re-opens the video.

**Architecture:** A new `pipeline/steps/fused_refine.py` runs the whole CUDA hot path inside a **single Metaflow `@step`** so the in-memory crop cache never crosses a Metaflow artifact boundary (which would serialize ~2.4GB to disk). Within that one function it: (1) streams 4K frame batches through NVDEC + YOLO, eagerly warps **every** detection via `KorniaNormalizer` and stores the 750×1050 crop keyed by `detection_id`; (2) reuses the existing `novelty.run` and `track.run` unchanged (they only touch detection metadata); (3) reuses `refine.run` with a new optional `decoded_crops` parameter — when provided, refine skips decode, the laplacian scan, and the warp block, pre-filling `normalized_by_detection` from the cache, then runs its existing scoring / glare / pHash / canonical-selection / ReID logic untouched. The non-CUDA (`docaligner`/MPS dev) path keeps the existing `detect → novelty → track → refine` stages byte-for-byte; the flow branches on `ctx.detector == "cuda"`.

**Tech Stack:** Python 3, Metaflow, decord (NVDEC), Kornia/torch (GPU warp), Ultralytics YOLOv8-OBB, OpenCV, pytest.

**Design decisions locked in with the user:**
- **Cache contents:** eagerly-warped 750×1050 normalized crops (`detection_id → np.ndarray`), ~2.4GB RAM for ~1000 detections. Not bbox crops, not full frames.
- **Reuse vs rewrite:** reuse `novelty.run` / `track.run` / `refine.run` via a cache parameter; do not duplicate refine's logic.
- **Laplacian "find a sharper nearby frame" rescan (concern "b"):** intentionally dropped on the fused path. Canonical candidates come only from YOLO detection frames.

**Expected effect:** refine's `decode_frames_gpu` (16.2s) and `kornia_warp_batch` (1.8s) drop to ~0; detect gains the warp of all detections (~8–13s, partially hidden behind decode-wait where GPU compute is otherwise ~65% idle). Net wall-time win is modest on its own, but the redundant decode is eliminated and the structure unblocks a follow-up decode/YOLO prefetch overlap. Validate against `run_telemetry` on RunPod after deploy.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `pipeline/steps/refine.py` | Modify | Add optional `decoded_crops` param to `run()`; when set, skip decode + laplacian + warp and pre-fill `normalized_by_detection` from the cache. |
| `pipeline/steps/detect.py` | Modify | Add `crop_cache` out-param to `_run_cuda_inference` so the streaming loop eagerly warps each detection and stores the crop. |
| `pipeline/steps/fused_refine.py` | Create | One-step orchestrator for the CUDA path: streaming inference+warp → `novelty.run` → `track.run` → `refine.run(decoded_crops=...)`. Returns a `RefineOutput`. |
| `pipeline/card_capture_flow.py` | Modify | Branch `detect` step on `ctx.detector == "cuda"`; forward through `novelty`/`track`/`refine` when fused. |
| `tests/pipeline/test_refine_fused.py` | Create | refine fused-mode behavior: no decode, cache used. |
| `tests/pipeline/test_detect_crop_cache.py` | Create | `_run_cuda_inference` populates the crop cache with one warped crop per detection. |
| `tests/pipeline/test_fused_refine.py` | Create | Orchestrator wires the sub-steps and returns a populated `RefineOutput`. |

---

## Task 1: refine accepts a pre-warped crop cache

**Files:**
- Modify: `pipeline/steps/refine.py:99` (the `run` signature) and the decode/laplacian/warp blocks at `pipeline/steps/refine.py:140-348`
- Test: `tests/pipeline/test_refine_fused.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_refine_fused.py
"""Fused-mode refine: when given a pre-warped crop cache, it must not decode."""
import numpy as np
import pytest


def _make_ctx(tmp_path):
    from pipeline.steps.start import RunContext
    crops = tmp_path / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    return RunContext(
        video_path="/nonexistent/video.MOV",
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "cards.sqlite"),
        detector="cuda",
        config_preset="balanced",
        crops_dir=str(crops),
        frame_dir=str(tmp_path / "frames"),
        use_kornia=False,          # force CPU-safe path; cache is pre-warped anyway
        corner_refinement=False,
        video_id=1,
    )


def _make_track_out():
    from pipeline.steps.track import TrackOutput
    corners = [[0.0, 0.0], [100.0, 0.0], [100.0, 140.0], [0.0, 140.0]]
    candidate = {
        "detection_id": 0,
        "frame_index": 5,
        "timestamp_ms": 166,
        "image_path": "",
        "confidence": 0.9,
        "score_total": 0.9,
        "score_components": {},
        "corners": corners,
    }
    track = {
        "instance_id": "inst-aaaaaaaa",
        "track_id": 1,
        "angle": 0.0,
        "candidate_detection_ids": [0],
        "first_frame_index": 5,
        "session_id": 1,
        "candidates": [candidate],
        "reid_embedding": None,
    }
    det_row = {
        "detection_id": 0, "frame_index": 5, "timestamp_ms": 166,
        "width": 3840, "height": 2160, "corners": corners,
        "confidence": 0.9, "source_frame_path": "", "triage_metrics": {},
        "novelty_score": 1.0,
    }
    return TrackOutput(
        tracks_data=[track],
        frame_to_session={"5": 1},
        tracker_events=[],
        detection_rows=[det_row],
        sampler_telemetry={"last_source_fps": 30.0},
        bg_model_path=None,
        accepted_frame_presence=[(5, 166, True)],
        frame_count=1,
        accepted_frame_count=1,
        video_id=1,
    )


def test_refine_fused_does_not_decode(tmp_path, monkeypatch):
    from pipeline.steps import refine
    from card_capture.storage import Storage
    Storage(tmp_path / "cards.sqlite").initialize()

    # decode_frames_gpu must never be called in fused mode.
    import card_capture.pipeline_utils as pu
    monkeypatch.setattr(
        pu, "decode_frames_gpu",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("decode_frames_gpu called in fused mode")),
    )

    # A pre-warped 750x1050 BGR crop keyed by detection_id.
    crop = np.random.randint(0, 256, (1050, 750, 3), dtype=np.uint8)
    decoded_crops = {0: crop}

    ctx = _make_ctx(tmp_path)
    track_out = _make_track_out()

    out = refine.run(ctx, track_out, decoded_crops=decoded_crops)

    assert len(out.refined_tracks) == 1
    entries = out.refined_tracks[0]["frame_entries"]
    assert len(entries) == 1
    # The cached crop was scored and written to disk (not re-warped from video).
    assert entries[0]["image_path"].endswith("_rectified.jpg")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/pipeline/test_refine_fused.py -q`
Expected: FAIL — `run() got an unexpected keyword argument 'decoded_crops'`.

- [ ] **Step 3: Add the `decoded_crops` parameter and the fused-mode guards**

In `pipeline/steps/refine.py`, change the signature at line 99:

```python
def run(ctx: RunContext, track_out: TrackOutput,
        decoded_crops: Optional[Dict[int, "Any"]] = None) -> RefineOutput:
```

Replace the decode block (current lines 166-174) with a guard:

```python
    # Compute union of all frame indices needed by Laplacian scan and Kornia warp,
    # then decode once via NVDEC instead of two separate CPU VideoCapture passes.
    # FUSED MODE: when decoded_crops is supplied, every detection was already
    # warped during the detect pass — skip decode and the laplacian rescan.
    decoded_images: Dict[int, np.ndarray] = {}
    _lap_scan_indices: set = set()
    _decode_frames_elapsed = 0.0
    if decoded_crops is None:
        _lap_scan_indices = _compute_laplacian_scan_indices(_lap_ranges, ctx.laplacian_scan_stride)
        _all_needed = canonical_indices | _lap_scan_indices
        _t_decode_start = time.time()
        if _all_needed:
            decoded_images = decode_frames_gpu(video_path, sorted(_all_needed))
        _decode_frames_elapsed = time.time() - _t_decode_start
```

Replace the laplacian scan block (current lines 176-195) with a guard:

```python
    _lap_results: Dict[str, list] = {}
    _laplacian_scan_elapsed = 0.0
    if decoded_crops is None:
        _t_lap_start = time.time()
        try:
            _lap_results = _laplacian_select_frames(
                video_path,
                _lap_ranges,
                scan_stride=ctx.laplacian_scan_stride,
                top_k=_lap_top_k,
                max_corner_gap=ctx.max_corner_gap_frames,
                decoded_frames=decoded_images if decoded_images else None,
            )
        except Exception as _e:
            print(f"[Refine] Laplacian scan failed, falling back to temporal-stride frames: {_e}")
        _laplacian_scan_elapsed = time.time() - _t_lap_start

        # Add any non-YOLO Laplacian-selected frames to the canonical decode set
        for _sel_list in _lap_results.values():
            for _fi, _ in _sel_list:
                canonical_indices.add(int(_fi))
    # ----------------------------------------
```

Guard corner refinement (current lines 302-312) so it only runs with real frames:

```python
        if ctx.corner_refinement and decoded_crops is None:
```

Replace the Kornia warp block (current lines 314-348) so fused mode pulls from the cache instead of warping:

```python
        # Batch Kornia warp if available. FUSED MODE: detections were already
        # warped during detect — pull the cached 750x1050 crops by detection_id
        # and skip the GPU warp entirely.
        normalized_by_detection: Dict[int, np.ndarray] = {}
        if decoded_crops is not None:
            for c in scored_candidates:
                cached = decoded_crops.get(c["detection_id"])
                if cached is not None:
                    normalized_by_detection[c["detection_id"]] = cached
        elif kornia_normalizer is not None and scored_candidates:
            batch_items = []
            batch_ids = []
            for c in scored_candidates:
                raw = decoded_images.get(c["frame_index"])
                if raw is None:
                    h = detection_lookup.get(c["detection_id"], {}).get("height", 10)
                    w = detection_lookup.get(c["detection_id"], {}).get("width", 10)
                    raw = np.zeros((h, w, 3), dtype=np.uint8)
                batch_items.append((raw, c["corners"]))
                batch_ids.append(c["detection_id"])
            if batch_items:
                try:
                    memory_before = _torch_cuda_memory_snapshot(ctx.kornia_device)
                    _t = time.time()
                    warped = kornia_normalizer.warp_canonical_batch(batch_items, rotate_180=ctx.rotate_180)
                    memory_after = _torch_cuda_memory_snapshot(ctx.kornia_device)
                    _elapsed = time.time() - _t
                    _t_ops["kornia_warp_batch"] += _elapsed
                    _op_counts["kornia_warp_batch"] += 1
                    _kornia_batches.append(_describe_kornia_batch(
                        batch_items,
                        batch_ids=batch_ids,
                        elapsed_s=_elapsed,
                        device=ctx.kornia_device,
                        memory_before=memory_before,
                        memory_after=memory_after,
                    ))
                    for did, img in zip(batch_ids, warped):
                        normalized_by_detection[did] = img
                except Exception as e:
                    print(f"Kornia warp failed: {e}")
                    normalized_by_detection = {}
```

Note: the per-candidate fallback at current lines 351-363 (`normalized = normalized_by_detection.get(...)` → `PrecisionNormalizer` if `None`) stays unchanged. In fused mode every candidate's `detection_id` is in the cache, so the fallback never fires; the `raw` placeholder zeros it builds are harmless because `bg_model` is `None` on the CUDA path (so `quad_novelty` is skipped).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/pipeline/test_refine_fused.py -q`
Expected: PASS.

- [ ] **Step 5: Run the existing refine telemetry test to confirm no regression**

Run: `python3 -m pytest tests/pipeline/test_refine_telemetry.py tests/test_pipeline_utils_gpu.py -q`
Expected: PASS (the non-fused path is unchanged; `decoded_crops` defaults to `None`).

- [ ] **Step 6: Commit**

```bash
git add pipeline/steps/refine.py tests/pipeline/test_refine_fused.py
git commit -m "feat(refine): accept pre-warped crop cache to skip redundant decode

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: detect's CUDA inference eagerly warps and caches every detection

**Files:**
- Modify: `pipeline/steps/detect.py:200-291` (`_run_cuda_inference`)
- Test: `tests/pipeline/test_detect_crop_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_detect_crop_cache.py
"""_run_cuda_inference fills a crop cache with one warped crop per detection."""
import numpy as np
from unittest.mock import MagicMock


def _make_ctx(tmp_path):
    from pipeline.steps.start import RunContext
    return RunContext(
        video_path="/nonexistent/video.MOV",
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "cards.sqlite"),
        detector="cuda",
        config_preset="balanced",
        crops_dir=str(tmp_path / "crops"),
        frame_dir=str(tmp_path / "frames"),
        rotate_180=False,
        kornia_device="cpu",
        video_id=1,
    )


def test_run_cuda_inference_populates_crop_cache(tmp_path, monkeypatch):
    from pipeline.steps import detect
    from card_capture.models import FrameSample, CornerDetection, FramePacket

    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 140.0), (0.0, 140.0)]

    sampler = MagicMock()
    sampler.last_selected_frame_count = 1
    sampler.last_source_fps = 30.0
    sampler.sample_batches.return_value = iter([[
        FrameSample(frame_index=5, timestamp_ms=166, image=frame, width=3840, height=2160),
    ]])

    detector = MagicMock()
    detector.confidence_threshold = 0.5

    def _detect_batch(packets, conf):
        out = []
        for p in packets:
            p.corner_detection = CornerDetection(corners=corners, confidence=0.9)
            out.append(p)
        return out
    detector.detect_batch.side_effect = _detect_batch

    # Stub the warp so the test needs no GPU: return a sentinel crop per call.
    class _FakeNorm:
        def warp_canonical_batch(self, batch_items, rotate_180=False):
            return [np.full((1050, 750, 3), i + 1, dtype=np.uint8) for i in range(len(batch_items))]
    monkeypatch.setattr(detect, "KorniaNormalizer", lambda **k: _FakeNorm())

    ctx = _make_ctx(tmp_path)
    crop_cache: dict = {}
    out = detect._run_cuda_inference(ctx, sampler, detector, tmp_path, tmp_path, crop_cache=crop_cache)

    assert len(out.detection_rows) == 1
    det_id = out.detection_rows[0]["detection_id"]
    assert det_id in crop_cache
    assert crop_cache[det_id].shape == (1050, 750, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/pipeline/test_detect_crop_cache.py -q`
Expected: FAIL — `KorniaNormalizer` is not imported in `detect`, and `_run_cuda_inference` has no `crop_cache` parameter.

- [ ] **Step 3: Add the import and the warp-and-cache logic**

At the top of `pipeline/steps/detect.py`, in the function-local imports area used by the CUDA path, add a module-level import after the existing imports near the top of the file:

```python
from card_capture.gpu_refinement import KorniaNormalizer
```

Change the `_run_cuda_inference` signature at line 200:

```python
def _run_cuda_inference(
    ctx: RunContext,
    sampler: "CudaSampler",
    detector: "CardcaptorUltralyticsDetector",
    output_dir: Path,
    frame_dir: Path,
    crop_cache: Optional[dict] = None,
) -> "DetectOutput":
```

Inside `_run_cuda_inference`, before the `for batch in sampler.sample_batches(...)` loop (after line 222), build a normalizer when caching is requested:

```python
    normalizer = None
    if crop_cache is not None:
        try:
            normalizer = KorniaNormalizer(width=750, height=1050, device=ctx.kornia_device)
        except Exception as e:
            print(f"[detect] KorniaNormalizer unavailable, crop cache disabled: {e}", flush=True)
            crop_cache = None
```

Inside the `for pkt in packets_out:` loop (current lines 251-266), after `detection_rows.append({...})` and before `det_id += 1`, collect the frame+corners for a per-batch warp. Restructure the batch body so that after building `detection_rows` you warp the whole batch at once. Replace the block at lines 248-266 with:

```python
        for f in batch:
            accepted_frame_presence.append((f.frame_index, f.timestamp_ms, True))

        # Map frame_index -> source image so we can warp detections from the
        # frame that is still in RAM (no second decode in refine).
        frame_by_index = {f.frame_index: f.image for f in batch}

        warp_items = []   # (image, corners)
        warp_ids = []     # detection_id aligned with warp_items
        for pkt in packets_out:
            cd = pkt.corner_detection
            this_id = det_id
            detection_rows.append(
                {
                    "detection_id": this_id,
                    "frame_index": pkt.frame_index,
                    "timestamp_ms": pkt.timestamp_ms,
                    "width": pkt.width,
                    "height": pkt.height,
                    "corners": [(float(p[0]), float(p[1])) for p in cd.corners],
                    "confidence": float(cd.confidence),
                    "source_frame_path": "",
                    "triage_metrics": {},
                }
            )
            if crop_cache is not None and cd.corners:
                src = frame_by_index.get(pkt.frame_index)
                if src is not None:
                    warp_items.append((src, [(float(p[0]), float(p[1])) for p in cd.corners]))
                    warp_ids.append(this_id)
            det_id += 1

        if crop_cache is not None and warp_items:
            try:
                warped = normalizer.warp_canonical_batch(warp_items, rotate_180=ctx.rotate_180)
                for wid, img in zip(warp_ids, warped):
                    crop_cache[wid] = img
            except Exception as e:
                print(f"[detect] crop warp failed for batch: {e}", flush=True)
```

Ensure `Optional` is imported in `detect.py` (it is used elsewhere; if not, add `from typing import Optional`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/pipeline/test_detect_crop_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm the non-cached call still works**

Run: `python3 -m pytest tests/pipeline/ -q -k "detect"`
Expected: PASS — `crop_cache` defaults to `None`, so existing callers (which pass no cache) get the original behavior with no warp.

- [ ] **Step 6: Commit**

```bash
git add pipeline/steps/detect.py tests/pipeline/test_detect_crop_cache.py
git commit -m "feat(detect): eagerly warp+cache detections during CUDA decode pass

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: fused_refine orchestrator (single Metaflow step body)

**Files:**
- Create: `pipeline/steps/fused_refine.py`
- Test: `tests/pipeline/test_fused_refine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_fused_refine.py
"""fused_refine.run wires inference->novelty->track->refine and returns a RefineOutput."""
from unittest.mock import MagicMock


def test_fused_refine_pipes_crop_cache_into_refine(tmp_path, monkeypatch):
    from pipeline.steps import fused_refine
    from pipeline.steps.start import RunContext
    from pipeline.steps.detect import DetectOutput
    from pipeline.steps.novelty import NoveltyOutput
    from pipeline.steps.track import TrackOutput
    from pipeline.steps.refine import RefineOutput

    ctx = RunContext(
        video_path="/nonexistent/video.MOV", output_dir=str(tmp_path),
        db_path=str(tmp_path / "cards.sqlite"), detector="cuda",
        config_preset="balanced", crops_dir=str(tmp_path / "crops"),
        frame_dir=str(tmp_path / "frames"), video_id=1,
    )

    # Stub sampler/detector construction.
    monkeypatch.setattr(fused_refine, "_build_sampler_detector", lambda c: (MagicMock(), MagicMock()))

    captured = {}

    def _fake_cuda_inference(c, s, d, od, fd, crop_cache=None):
        crop_cache[0] = "CROP0"   # pretend one detection was warped
        return DetectOutput(
            frame_count=1, accepted_frame_count=1,
            accepted_frame_presence=[(5, 166, True)],
            detection_rows=[{"detection_id": 0, "frame_index": 5, "corners": [],
                             "confidence": 0.9, "timestamp_ms": 166, "width": 3840,
                             "height": 2160, "source_frame_path": "", "triage_metrics": {}}],
            sampler_telemetry={"sampler_type": "CudaSampler"},
            video_id=1, detect_telemetry={"yolo_frames": 1},
        )
    monkeypatch.setattr(fused_refine, "_run_cuda_inference", _fake_cuda_inference)

    monkeypatch.setattr(
        fused_refine.novelty, "run",
        lambda c, det: NoveltyOutput(
            detection_rows=det.detection_rows, sampler_telemetry=det.sampler_telemetry,
            bg_model_path=None, accepted_frame_presence=det.accepted_frame_presence,
            frame_count=1, accepted_frame_count=1, video_id=1),
    )
    monkeypatch.setattr(
        fused_refine.track, "run",
        lambda c, nov: TrackOutput(
            tracks_data=[], frame_to_session={}, tracker_events=[],
            detection_rows=nov.detection_rows, sampler_telemetry=nov.sampler_telemetry,
            bg_model_path=None, accepted_frame_presence=nov.accepted_frame_presence,
            frame_count=1, accepted_frame_count=1, video_id=1),
    )

    def _fake_refine_run(c, trk, decoded_crops=None):
        captured["decoded_crops"] = decoded_crops
        return RefineOutput(
            refined_tracks=[], tracks_data=trk.tracks_data,
            detection_rows=trk.detection_rows, sampler_telemetry=trk.sampler_telemetry,
            bg_model_path=None, tracker_events=[], accepted_frame_presence=trk.accepted_frame_presence,
            frame_count=1, accepted_frame_count=1, video_id=1)
    monkeypatch.setattr(fused_refine.refine, "run", _fake_refine_run)

    out = fused_refine.run(ctx)

    assert isinstance(out, RefineOutput)
    # The crop cache built during inference was threaded into refine.
    assert captured["decoded_crops"] == {0: "CROP0"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/pipeline/test_fused_refine.py -q`
Expected: FAIL — `No module named 'pipeline.steps.fused_refine'`.

- [ ] **Step 3: Create the orchestrator**

```python
# pipeline/steps/fused_refine.py
"""Fused CUDA hot path — decode + YOLO + eager warp + track + refine in one step.

Runs the entire CUDA detection/refinement path inside a single Metaflow @step so
the in-memory crop cache (~2.4GB of 750x1050 normalized crops) never crosses a
Metaflow artifact boundary. Stages 4 (novelty) and 5 (track) are reused unchanged
— they touch only detection metadata. refine.run is reused with the crop cache so
it never re-decodes the video.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from pipeline.steps import novelty, track, refine
from pipeline.steps.detect import _build_sampler_detector, _run_cuda_inference
from pipeline.steps.refine import RefineOutput
from pipeline.steps.start import RunContext


def run(ctx: RunContext) -> RefineOutput:
    """Execute the fused CUDA path and return a RefineOutput for the score step."""
    sampler, detector = _build_sampler_detector(ctx)

    crop_cache: Dict[int, Any] = {}
    _t_infer = time.time()
    detect_out = _run_cuda_inference(
        ctx, sampler, detector, ctx.output_dir, ctx.frame_dir, crop_cache=crop_cache,
    )
    _infer_elapsed = time.time() - _t_infer

    novelty_out = novelty.run(ctx, detect_out)
    track_out = track.run(ctx, novelty_out)
    refine_out = refine.run(ctx, track_out, decoded_crops=crop_cache)

    # Surface fused-path telemetry alongside refine's own op breakdown so the
    # handler diagnostic still shows where time went.
    existing = getattr(refine_out, "refine_telemetry", None) or {}
    existing.update({
        "fused": True,
        "fused_inference_s": round(_infer_elapsed, 3),
        "crops_cached": len(crop_cache),
        "detect_telemetry": detect_out.detect_telemetry,
    })
    refine_out.refine_telemetry = existing  # type: ignore[attr-defined]
    return refine_out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/pipeline/test_fused_refine.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/steps/fused_refine.py tests/pipeline/test_fused_refine.py
git commit -m "feat(pipeline): add fused_refine single-step CUDA orchestrator

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: wire the flow to branch on the CUDA path

**Files:**
- Modify: `pipeline/card_capture_flow.py:77-126`

- [ ] **Step 1: Write the failing test**

```python
# tests/pipeline/test_flow_fused_branch.py
"""The flow uses fused_refine for the cuda detector and forwards downstream."""


def test_flow_imports_fused_refine():
    import pipeline.card_capture_flow as flow
    # fused_refine must be importable by the flow module.
    assert hasattr(flow, "fused_refine")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/pipeline/test_flow_fused_branch.py -q`
Expected: FAIL — `flow` has no attribute `fused_refine`.

- [ ] **Step 3: Add the branch and forwarding**

In `pipeline/card_capture_flow.py`, extend the import at line 12:

```python
from pipeline.steps import (
    detect, novelty, track, refine, score, resolve, fuse, dedup, store,
    fused_refine,
)
```

Replace the `detect` step (lines 77-91) with a branch that sets `self._fused`:

```python
    @step
    def detect(self):
        _t0 = _time.time()
        ctx = self.run_context
        _run_id = self.ui_run_id or current.run_id
        if ctx.detector == "cuda":
            # Fused path: decode + warp + novelty + track + refine in one step
            # so the crop cache stays in memory (no Metaflow artifact spill).
            self.refine_out = fused_refine.run(ctx)
            self._fused = True
            _elapsed_ms = int((_time.time() - _t0) * 1000)
            _record_stage_timing(
                ctx.db_path, _run_id, ctx.video_id or 0, "detect", _elapsed_ms,
                extra=getattr(self.refine_out, "refine_telemetry", None) or {},
            )
        else:
            self.detect_out = detect.run(ctx)
            self._fused = False
            _elapsed_ms = int((_time.time() - _t0) * 1000)
            _record_stage_timing(
                ctx.db_path, _run_id, ctx.video_id or 0, "detect", _elapsed_ms,
                extra=self.detect_out.detect_telemetry or {},
            )
        self.next(self.novelty)
```

Replace `novelty` (lines 93-100):

```python
    @step
    def novelty(self):
        if not self._fused:
            _t0 = _time.time()
            self.novelty_out = novelty.run(self.run_context, self.detect_out)
            _elapsed_ms = int((_time.time() - _t0) * 1000)
            ctx = self.run_context
            _record_stage_timing(ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0, "novelty", _elapsed_ms)
        self.next(self.track)
```

Replace `track` (lines 102-109):

```python
    @step
    def track(self):
        if not self._fused:
            _t0 = _time.time()
            self.track_out = track.run(self.run_context, self.novelty_out)
            _elapsed_ms = int((_time.time() - _t0) * 1000)
            ctx = self.run_context
            _record_stage_timing(ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0, "track", _elapsed_ms)
        self.next(self.refine)
```

Replace `refine` (lines 111-126):

```python
    @step
    def refine(self):
        if not self._fused:
            _t0 = _time.time()
            self.refine_out = refine.run(self.run_context, self.track_out)
            _elapsed_ms = int((_time.time() - _t0) * 1000)
            ctx = self.run_context
            _refine_extra = getattr(self.refine_out, "refine_telemetry", None) or {}
            _record_stage_timing(
                ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0,
                "refine", _elapsed_ms, extra=_refine_extra,
            )
        else:
            # Fused path already produced self.refine_out in the detect step;
            # emit a refine stage event so the diagnostic still shows it.
            ctx = self.run_context
            _refine_extra = getattr(self.refine_out, "refine_telemetry", None) or {}
            _record_stage_timing(
                ctx.db_path, self.ui_run_id or current.run_id, ctx.video_id or 0,
                "refine", 0, extra=_refine_extra,
            )
        self.next(self.score)
```

`score` (line 128) is unchanged — it already reads `self.refine_out`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/pipeline/test_flow_fused_branch.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm the flow still compiles as a Metaflow graph**

Run: `python3 pipeline/card_capture_flow.py --no-pylint show`
Expected: prints the step graph (`start → detect → novelty → track → refine → score → resolve → fuse_fanout → fuse → fuse_join → dedup → store → end`) with no import or syntax error.

- [ ] **Step 6: Commit**

```bash
git add pipeline/card_capture_flow.py tests/pipeline/test_flow_fused_branch.py
git commit -m "feat(flow): route the cuda detector through fused_refine, forward downstream

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: full regression + RunPod verification

**Files:** none (verification only)

- [ ] **Step 1: Run the unit suite (skip the slow integration test)**

Run: `python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py`
Expected: no *new* failures beyond the documented pre-existing ones (CLAUDE.md lists `tests/migrations/test_schema.py::test_migrations_are_idempotent`, several in `test_wave1/2_robustness.py`, and `test_path_equivalence.py`). Record the failure list and diff it against that baseline.

- [ ] **Step 2: Verify the non-CUDA path is untouched**

Run: `python3 -m pytest tests/pipeline/ -q -k "not fused and not crop_cache"`
Expected: PASS — confirms `detect → novelty → track → refine` behavior is unchanged for `docaligner`/`fake`/MPS.

- [ ] **Step 3: Commit any test-baseline notes (if a doc tracks them)**

No code change expected here. If `docs/` tracks the known-failure baseline, update it; otherwise skip.

- [ ] **Step 4: Deploy + RunPod telemetry check (manual, after merge to origin/main)**

`docker/start.sh` syncs the worker to `origin/main` on container start, so push to main is the deploy. After a RunPod run completes, pull the latest `card_capture_output/run_*/run_*_handler_output.json` and confirm in `diagnostics.stage_payloads`:
- `stage_detect.elapsed_ms` now includes the warp time and `refine_telemetry.fused == true`.
- `stage_refine.op_seconds.decode_frames_gpu` is absent or ~0 (no second decode).
- `stage_refine.op_seconds.kornia_warp_batch` is absent or ~0 (warp moved to detect).
- `diagnostics.card_instances_this_run` and `card_views_total` are within ~10% of the pre-change baseline run (`run_f2037cad`: 18 instances / 140 views) — i.e. dropping the laplacian rescan did not materially change output counts.

---

## Self-Review

**Spec coverage:**
- Collapse stages 3–6 into one in-memory step → Task 3 (`fused_refine`) + Task 4 (flow branch). ✅
- Eager-warp every detection, cache by `detection_id` → Task 2. ✅
- Reuse refine via a cache param → Task 1. ✅
- Ignore the laplacian rescan on the fused path → Task 1 guards skip `_laplacian_select_frames` when `decoded_crops` is set. ✅
- Keep the non-CUDA path intact → Task 4 branches on `ctx.detector == "cuda"`; defaults (`decoded_crops=None`, `crop_cache=None`) preserve old behavior; Task 5 Step 2 verifies. ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✅

**Type consistency:**
- `decoded_crops: Dict[int, np.ndarray]` (detection_id → 750×1050 BGR uint8) is produced in Task 2 (`crop_cache[wid] = img`), threaded in Task 3 (`decoded_crops=crop_cache`), consumed in Task 1 (`decoded_crops.get(c["detection_id"])`). Keys match (`detection_id`). ✅
- `_run_cuda_inference(..., crop_cache=None)` signature in Task 2 matches the call in Task 3. ✅
- `fused_refine.run(ctx) -> RefineOutput` matches what the flow assigns to `self.refine_out` and what `score.run` consumes. ✅
- `refine_telemetry` attribute set via `setattr` in both refine and fused_refine, read by the flow — consistent with the existing pattern. ✅

**Risk note:** the one behavioral change is dropping the dense laplacian sharpness rescan on the CUDA path (canonical frames now come only from YOLO detection frames). Task 5 Step 4 explicitly checks output counts against the baseline to catch a quality regression.
