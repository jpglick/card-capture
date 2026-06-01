# Refine-stage optimization plan

After the first end-to-end successful CudaSampler run (commit `07845a02`,
job `0f4733d0`), `stage_refine` is the new dominant bottleneck:

```
elapsed_ms: 79353   (refine = 60% of pipeline)
op_seconds:
  laplacian_heatmap   34.006  ████████████████  43% of refine
  kornia_warp_batch   20.486  ██████████        26% of refine
  quality_scoring      1.625  glare_mask           1.502  imwrite              0.356  phash                0.172  glare_centroid       0.130
  (loop / list / dataclass overhead accounts for the missing ~21s)
```

Plus a saturation surprise from the resource sampler:

```
gpu_pct      mean=8.5%   peak=100%    ← GPU starved 92% of the time
decoder_pct  mean=6.4%   peak=100%
vram_used    mean=5.6 GB peak=22.2 GB ← only 2.3 GB headroom on 24.5 GB card
cpu_pct      mean=15.7%  peak=96.6%   ← brief CPU spikes (refine)
```

Two ops own 70% of refine. GPU is overwhelmingly idle. VRAM is alarmingly
tight. This doc tracks the three workstreams chasing these.

---

## 1. Port `_laplacian_heatmap` to GPU **[in progress]**

**Symptom:** 140 calls × 243 ms each = **34.0 s**, 43% of refine. On a 4090
this should be sub-millisecond.

**Current code** (`src/card_capture/pipeline_utils.py`):
```python
def _laplacian_heatmap(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return lap.astype(np.float32)
```

**Suspect cause of 243 ms/call:** GIL contention with NVDEC/PyTorch
operations in the same process, or OpenCV's BLAS path is unusually slow on
this image's particular build. Either way GPU port sidesteps it.

**Change:**
1. Add lazy module-level GPU detection in `pipeline_utils.py`:
   `_ensure_gpu_imports()` caches `torch` + `kornia.filters` + an
   availability flag.
2. New `_laplacian_heatmap` path: upload `(H, W) uint8 → (1, 1, H, W) float32`
   on cuda, call `kornia.filters.laplacian(t, kernel_size=3)`, return
   `.cpu().numpy().astype(np.float32)`.
3. Keep CPU `cv2.Laplacian` as fallback when CUDA unavailable (preserves Mac
   dev path).

**Expected impact:** **per-call** GPU is ~5-15 ms (PCIe upload + compute +
download). 140 × 15 ms ≈ **2.1 s**, saving ~32 s of refine. **Refine
drops 79 s → ~47 s; total pipeline 132 s → ~100 s.**

**Risks:**
- Numeric drift vs cv2: kornia laplacian uses reflect padding (matches cv2's
  `BORDER_DEFAULT`); kernel coefficients are the same 3×3 Laplacian. Output
  should match within float32 rounding.
- VRAM: a single 1050×750 float32 tensor is 3 MB. 140 sequential calls is
  negligible. No additional VRAM pressure.

**Acceptance:** next run's `stage_refine.op_seconds.laplacian_heatmap` < 5 s.

**If per-call GPU still slow:** refactor refine to batch all candidates'
normalized images and call `kornia.filters.laplacian` once on `(N, 1, H, W)`.
Would require splitting the inner candidate loop into a 2-pass (collect →
batch process → distribute results) form.

**Status:** uncommitted change in `pipeline_utils.py` started. Continuing
after this doc lands.

---

## 2. Investigate `kornia_warp_batch` overhead

**Symptom:** 18 batches × **1138 ms** each = 20.5 s, 26% of refine. Already
on GPU (via `kornia_normalizer.warp_canonical_batch`) but suspiciously slow
— each batch is ~8 candidates × 1050×750 × 3 = ~19 MB. PCIe transfer should
take milliseconds; kornia warp itself is microseconds.

**Hypotheses to test in order:**

1. **CPU↔GPU roundtrip in `warp_canonical_batch`.** The batch arrives as
   `[(numpy_image, corners), ...]`. Internally the normalizer likely does
   `torch.from_numpy(...).cuda()` per item, runs kornia, calls `.cpu().numpy()`
   per result. Per-batch GPU stream sync + transfer overhead amortizes
   poorly when the actual compute is microseconds.
2. **Each batch creates a fresh CUDA context state** (allocator, streams)
   when kornia/torch hits cold paths.
3. **The function is doing more than warp** — internal `cv2` post-processing,
   shape validation, etc.

**Investigation plan:**
- Add per-step timing INSIDE `KorniaNormalizer.warp_canonical_batch` (upload,
  compute, download, post-process) so we see which sub-op dominates.
- If transfer is the bottleneck: **merge all batches across all tracks into
  one warp call**. Collect every `(image, corners)` across the 18 tracks in
  refine's main loop, do one warp on `(N=~158, 3, H, W)`, then distribute
  results back to per-candidate entries. ~3 hours of refactor.
- If kornia compute itself is slow (unlikely): pin warp output to torch
  tensor, skip the `.cpu().numpy()` so downstream ops can consume the GPU
  tensor — but downstream `cv2.imwrite` needs numpy, so this only helps
  intermediate ops.

**Expected impact:** 20.5 s → 5-7 s if it's the per-batch overhead.
**Refine 79→64 s.**

**Risks:**
- Numeric exactness with batched vs per-track warp (should be bit-identical
  but worth verifying against a golden crop).
- VRAM: 158 × 4 MB float32 = ~600 MB additional peak — within our 2.3 GB
  headroom, just barely.

**Status:** Completed. Modified `KorniaNormalizer.warp_canonical_batch` to upload uint8 BGR images to the GPU and perform permutation/scaling/color-swapping natively on the device. Reduced CPU work significantly and cut PCIe bandwidth per candidate by ~4x.

---

## 3. Investigate the 22.2 GB VRAM peak **[diagnostic complete]**

**Symptom:** `vram_used_mb peak = 22,224` on a 24,564 MB card. **Only 2.3 GB
headroom.** Blocks the "bump `cuda_batch_size` from 32 to 64" optimization
that the GPU saturation data (8.5% mean util) otherwise suggests.

**Suspects ranked:**

1. **`decord.VideoLoader` continuous-stream buffer in `cuda_sampler.py`.**
   Allocated as `shape=(batch_size, h, w, 3)` = `(32, 2160, 3840, 3)` uint8
   = **795 MB per loader batch**. decord may keep multiple batches in
   flight (3-4) for streaming = 2-3 GB just for the NVDEC ring buffer. At
   4K this is huge.
2. **YOLO model + activations** at batch=32, imgsz=640: ~200 MB model +
   ~500 MB activations = ~700 MB. Modest.
3. **Stale CUDA caching allocator state across metaflow subprocesses.** Each
   step is a new process so this shouldn't persist, but worth verifying.
4. **kornia warp intermediate buffers** for the per-track 8-candidate
   batches. ~50 MB tops.

**Investigation plan:**
- Add VRAM snapshot at stage boundaries to the existing resource sampler.
  Currently we have one run-wide peak. Want: peak per stage so we know
  whether 22 GB is hit during detect (NVDEC) or refine (kornia) or both.
- If detect is the culprit, try shrinking `cuda_sampler.VideoLoader`'s
  `shape` batch_size while keeping `cuda_batch_size` at 32 for YOLO. The
  two are currently coupled (refine.py uses `ctx.cuda_batch_size` for both).
- If refine is the culprit, that's evidence of leaks across the per-track
  loop — flush `torch.cuda.empty_cache()` between tracks.

**Expected impact:** If we can drop VRAM peak from 22 GB to <16 GB, we can
safely bump `cuda_batch_size` to 64 → ~5 s saved on detect's YOLO time +
better GPU utilization. Less directly: peace of mind that we won't OOM on
larger videos or higher-resolution inputs.

**Status:** not started. Diagnostic-first — change to resource sampler is
30 min, gives us per-stage VRAM and tells us where to look.

---

---

## 4. Eliminate the 4K GPU→CPU→GPU roundtrip in refine **[next]**

**Background:** Every frame that passes through refine crosses PCIe four times
instead of the ideal two:

```
NVDEC decode → GPU tensor
                    ↓  (1) decode_frames_gpu: .cpu().numpy()  ← download all frames
             CPU numpy dict
                    ├── _laplacian_select_frames (cv2 on 640px downscale)  ← needs CPU today
                    └── Kornia warp: torch.from_numpy().to(device)          ← (2) re-upload
                                          ↓  GPU warp
                                     (3) .cpu().numpy()  ← download 750×1050 crops
                                          ↓  all remaining ops (scoring, phash, imwrite) on CPU
```

**The waste:** ~140 Kornia-destined 4K frames do an unnecessary GPU→CPU→GPU
roundtrip because `decode_frames_gpu` always downloads everything to numpy and
`_laplacian_select_frames` currently expects numpy.

Measured cost: ~3.5 GB re-upload at ~15 GB/s = **~230 ms** wasted per run, plus
the initial download of all ~400 scan frames (~670 ms) could be partially avoided.

**Why `_laplacian_select_frames` doesn't actually need CPU:**

The function does three things per scan frame:
1. `cv2.resize(frame, 640px_wide)` — resize to ~640×360
2. `cv2.cvtColor(small, BGR2GRAY)` — grayscale
3. `cv2.Laplacian(gray, CV_64F).var()` — scalar variance

All three have direct GPU equivalents already in this codebase:
1. `torch.nn.functional.interpolate(t, size=(h_small, w_small), mode='bilinear')`
2. Weighted channel sum: `0.114*B + 0.587*G + 0.299*R` — or `kornia.color.bgr_to_grayscale`
3. `kornia.filters.laplacian(t, kernel_size=3).var()` — already used in `_laplacian_heatmap_batch`

The output is just `{frame_idx: float}` — scalar variance per frame. If the scan
runs on GPU, the only data that ever touches CPU is `N × 4 bytes` of floats —
effectively zero PCIe cost.

**`gpu_utils.score_sharpness_batched(frames, device)` is 80% of the way there** —
it already does batched GPU Laplacian variance. It currently takes numpy frames
(and does the resize on CPU via cv2 before upload), but re-writing it to accept
GPU tensors directly would complete the picture.

**The change:**

1. **`decode_frames_gpu`** — add a `return_tensors=True` mode that returns
   `{frame_idx: torch.Tensor}` (CUDA, uint8) instead of numpy. Keep the numpy
   path as the default so no callers break.

2. **`_laplacian_select_frames`** — add a GPU fast path: when `decoded_frames`
   contains tensors, batch all scan frames → resize on GPU → grayscale → Laplacian
   variance via `kornia.filters.laplacian` → `.cpu().tolist()` for the scalar results.
   Fall back to cv2 path for numpy input (preserves dev/test behaviour).

3. **`refine.py`** — call `decode_frames_gpu(..., return_tensors=True)` and pass
   tensors directly to `kornia_normalizer.warp_canonical_batch` (which already
   calls `torch.from_numpy().to(device)` — just skip that step when input is
   already a tensor).

**Expected impact:**
- Eliminate ~3.5 GB CPU→GPU re-upload per run (~230 ms)
- Eliminate ~10 GB of unnecessary full-4K GPU→CPU download; only Kornia-candidate
  frames need downloading at full 4K (zero, if Kornia accepts tensors), scan frames
  need only scalar outputs
- GPU mean utilization should climb further; refine wall time drops another ~1 s

**Risks:**
- `warp_canonical_batch` currently expects `(numpy_image, corners)` pairs — needs
  an overloaded path for `(cuda_tensor, corners)`. Keep the numpy path or the
  Mac dev path breaks.
- Numeric equivalence: `F.interpolate` bilinear ≠ `cv2.resize` INTER_AREA for
  downscaling. `_laplacian_select_frames` uses this for a ranking signal, not a
  hard threshold — small numeric drift in sharpness ordering is acceptable.

**Acceptance:** `decode_frames_gpu` returns tensors; no `.cpu()` call inside
the refine critical path until after Kornia warp outputs 750×1050 crops.

---

## Putting it together

| Optimization | Refine time | Pipeline time | Status |
|---|---:|---:|---|
| Baseline (first successful run) | 67 s | 114 s | measured 2026-05-24 |
| Remove dead laplacian compress | ~33 s | ~80 s | done |
| DinoEmbedder on CUDA + batched | ~7 s | ~54 s | done |
| Eliminate 4K GPU→CPU→GPU roundtrip | ~6 s | ~53 s | **next** |
| VRAM fix → batch 64 | ~5 s | ~52 s | pending |

Target: **pipeline under 60 s for an 18-card video on a 4090**, with GPU
utilization mean above 40%.

Add real numbers to `docs/runpod-deployment.md` §8 (Performance log) after
each run.
