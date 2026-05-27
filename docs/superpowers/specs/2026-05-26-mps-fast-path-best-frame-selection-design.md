# MPS Fast Path — Best-Frame Selection Design

**Date:** 2026-05-26
**Status:** Approved (design); pending implementation plan
**Scope:** The local MPS / `docaligner` detection+refinement path on Apple Silicon. CUDA path is out of scope except where it shares code.

---

## Problem

A recent uncommitted refactor merged the MPS path into the CUDA-oriented "GPU-resident frames / thumbnail-download" fused pipeline. This caused a performance *and* quality regression versus a few iterations ago:

1. **Sampling swap.** `AdaptivePresenceSampler` (presence-gated, valley-split, ~3 fps effective) was replaced by `StrideSampler` at **15 fps** (`pipeline/steps/detect.py:178`). This is good for coverage but the path then does expensive high-res work on *every* sampled frame.
2. **CUDA-shaped execution on unified memory.** The fused loop (`pipeline/steps/detect.py:207`) uploads each 4K batch to the GPU, resizes to a 640px thumbnail, downloads the thumbnail to CPU for YOLO, **and eagerly Kornia-warps every detection** to 750×1050 into an in-memory crop cache (`detect.py:354-383`). The "only thumbnails cross the bus" idea is a PCIe optimization with no benefit on Apple unified memory, and the eager warp does the most expensive operation on every one of ~15 fps × all detections.
3. **Cache thrash.** `torch.mps.empty_cache()` every 5 batches (`detect.py:402-404`) forces a full GPU stall; `batch_size = 4` (`detect.py:258`) is tiny.
4. **Silent CPU fallbacks.** Device resolution can silently downgrade to CPU, and the warp can fall back to the CPU `cropper.py`. These have been a recurring source of "mysteriously slow" runs.

The user requirement: **keep dense stride sampling** (most frames contain a card; the old adaptive sampler missed the best frames), but **only do high-res work on the best frames per card** — where "best" means the card shown **as flat and clear as possible** — and make the route **MPS-or-fail** (no silent CPU fallback).

## Goals

- Retain `StrideSampler` dense coverage.
- Do the expensive Kornia warp on at most **5 frames per card**, not every detection.
- Decide the final "flat and clear" winner on the **real warped pixels**, one image per card.
- Make the MPS route **MPS-or-fail**: no silent CPU compute fallback anywhere in this path.
- Net: large reduction in high-res warps; removal of cache-thrash and fallback overhead.

## Non-Goals

- No change to the CUDA path's external behavior (it may share refactored helpers).
- No change to Front/Back resolution (Stage 8), dedup (Stage 10), or storage schema beyond feeding them one canonical crop per card.
- No learned ranker; flatness/clarity are deterministic heuristics.
- Novelty gate (Stage 4) is **not** removed in this change — left inert (see below).

---

## Design

### Single streaming pass (Stages 1–3) — nothing warped

There is **no separate pre-processing/scan pass**. One pass over the video:

1. **`StrideSampler` @ 15 fps** (kept) — dense, uniform coverage.
2. **GPU prep on MPS:** decode → upload → NV12→BGR → resize to 640px, kept on the GPU tensor.
3. **YOLO-OBB on MPS** → 4 corners + confidence per detection.
4. **Inline cheap score per detection** (no warp), stored in the detection's `triage_metrics`:
   - **Flatness** ∈ [0,1] — pure geometry on the 4 corners: closeness of the quad's aspect ratio to the 5:7 target (750:1050 ≈ 0.714) combined with rectangularity (interior angles near 90°, opposite sides near-equal). No pixels touched.
   - **Clarity** ∈ [0,1] — Laplacian-variance on the card's axis-aligned bbox ROI cropped from the full-res GPU tensor (grayscale Laplacian via a GPU conv), normalized. Cheap; same pass; far more reliable than a thumbnail-wide measure.
   - The full-res GPU tensor is released immediately after the ROI read. The eager per-detection warp and the in-memory crop cache are **removed**.

### Group into cards (Stages 4–5)

- **Stage 5 tracking** assigns object IDs (`TrackState.instance_id`) over the collected detection metadata. This is what makes "per card" meaningful — the tracker *is* the card identifier. No images needed.
- Empty/background frames produce no detections → no track. Handled for free.
- **Stage 4 novelty gate** is left inert (redundant once detection drives the path). Removing it is a separate follow-up.

### Select → warp → repick (Stage 6 refine)

1. **Shortlist:** per track, rank candidates by a combined cheap score `w_flat·flatness + w_clear·clarity` and take the **top 5**.
2. **Warp:** refine re-decodes only the **union of selected frame indices** (existing `decode_frames_gpu` machinery in `pipeline/steps/refine.py`) and GPU-warps those ≤5/card to 750×1050 via Kornia.
3. **Repick:** re-score each warped crop with `src/card_capture/scoring.py` (sharpness + flatness/aspect + glare, on the real warped pixels) and keep the **single best** crop per track as that card's canonical image.

### Fusion bypassed

Stage 9 (lighting-diverse fusion) is bypassed for this path. The best-of-5 warped crop is the canonical image fed to resolve/dedup/store where the fused canonical previously went.

### MPS-or-fail policy (no CPU compute fallbacks)

- Device resolution must yield `mps` on this route or **raise** with a clear message — no silent CPU downgrade.
- Kornia GPU warp is **required** or raise — no CPU `cropper.py` fallback in this path.
- Remove `CC_*_ALLOW_CPU_FALLBACK`-style silent CPU paths from the MPS route.
- Remove the per-5-batch `torch.mps.empty_cache()`.
- Raise MPS `batch_size` above 4.

> Note: passing YOLO a small **numpy** image is *not* a CPU-compute fallback — inference still runs on MPS; it is only where the resize/letterbox happens. The fallbacks being removed are the CPU *compute* ones.

---

## Components & Interfaces

| Unit | Responsibility | Input → Output |
|---|---|---|
| Flatness scorer | Quad geometry → flatness ∈ [0,1] | 4 corners → float |
| Clarity scorer | GPU ROI Laplacian variance → clarity ∈ [0,1] | full-res GPU tensor + bbox → float |
| Detect loop (`_run_fused_inference`, MPS branch) | Stream decode→YOLO→cheap score; **no warp** | video → detection_rows w/ `triage_metrics{flatness,clarity}` |
| Per-track shortlist | top-5 by combined cheap score | track candidates → ≤5 frame indices/track |
| Refine warp+repick | warp selected, re-score, keep best | selected indices → 1 canonical crop/track |

Each scorer is independently unit-testable and pure (geometry, or tensor+bbox → float).

---

## Telemetry

Record per run: frames sampled, detections, tracks, **high-res warps performed** (should be ≈ Σ min(5, candidates/track)), fused-inference wall time, refine wall time, resolved device. The warp count is the primary proof the optimization took effect.

---

## Testing

- **Unit:** flatness scorer (synthetic quads: perfect rectangle → ~1.0; skewed/keystoned → low). Clarity scorer (sharp vs Gaussian-blurred crop → higher vs lower). Top-5 selection (synthetic track with N>5 candidates → correct 5 by score).
- **MPS-or-fail:** with MPS forced unavailable, the route raises rather than running on CPU.
- **Integration:** run the user-provided test video before/after; compare wall-time, warp count (telemetry), cards detected/kept, and a visual check that chosen crops are flat and sharp.

---

## Cleanup

- Delete abandoned duplicate `pipeline/steps/detect_fused_helper.py`.
- Delete stray debug images: `debug_pipeline_thumb.jpg`, `diagnostic_worker_thumb.jpg`, `test_thumb_gpu.jpg`.

---

## Risks & Mitigations

- **Flattest frames all motion-blurred.** Mitigated by including clarity in the shortlist score (not flatness-only) and by 15 fps coverage giving many candidates.
- **Re-decode cost in refine.** Bounded — only the union of selected frames (≤5/card) is decoded, far fewer than all sampled frames.
- **Ultralytics tensor-vs-numpy input parity on MPS.** If feeding a preprocessed GPU tensor to the OBB model doesn't match numpy-input accuracy/letterboxing, keep the numpy thumbnail hand-off (not a CPU-compute fallback). The dominant win is removing the eager warp, not the thumbnail download.
- **Downstream coupling (resolve/dedup/store) expecting a fused canonical.** The per-track best crop is substituted at that boundary; verify these stages accept it unchanged.
```
