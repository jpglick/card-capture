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

**Status:** not started. Blocked on landing the laplacian port to free
context window for the deeper refactor.

---

## 3. Investigate the 22.2 GB VRAM peak

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

## Putting it together

Cumulative expected wins if all three land:

| Optimization | Refine time | Pipeline time |
|---|---:|---:|
| Current | 79 s | 132 s |
| + laplacian on GPU | ~47 s | ~100 s |
| + kornia warp single-batch | ~32 s | ~85 s |
| + VRAM fix → batch 64 | ~30 s | ~80 s |

Target: **pipeline under 90 s for an 18-card video on a 4090**, with GPU
utilization mean above 30%.

Add real numbers to `docs/runpod-deployment.md` §8 (Performance log) after
each run.
