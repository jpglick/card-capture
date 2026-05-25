# GPU Throughput: TensorRT YOLO + Decode/Inference Overlap + GPU-Batched Scoring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. The plan is three independently-revertible PHASES on one branch; ship/measure between phases.

**Goal:** Cut the fused CUDA pass from ~31.6s toward single digits by attacking the three real costs the telemetry exposed: YOLO compute (11.2s), the serial CPU scoring tail (~14.4s), and the serial decode→YOLO→warp loop (~6s). Target: YOLO ~3s (TensorRT), CPU tail ~3s (GPU-batched), decode hidden behind YOLO (overlap).

**Architecture:** Three phases on branch `gpu-throughput`. **Phase A** runs YOLO via a TensorRT engine (FP16), built on the worker GPU in the existing `start.sh` warmup and cached, with graceful fallback to FP16 `.pt` then FP32 `.pt`. **Phase B** adds an in-process prefetch thread so NVDEC decodes batch N+1 while the GPU runs YOLO+warp on batch N (the VRAM freed by the GPU-resident work — 87%→66% — makes this fit). **Phase C** keeps warped crops GPU-resident and computes ALL quality/glare/pHash components **batched on the GPU** across every view at once, eliminating the per-crop Python/OpenCV loop; the two ops without clean GPU equivalents (`cv2.connectedComponentsWithStats` for spatial-glare, `cv2.dct` for pHash) get documented GPU approximations.

**Tech Stack:** Ultralytics 8.4.x (`export(format="engine")`), TensorRT, torch + kornia (GPU image ops, DCT via matmul), decord NVDEC, Metaflow, pytest.

**Decisions locked with user:** one combined plan (3 phases, one branch); TensorRT (not FP16-only) with fallback; full-GPU scoring including approximations for the hard ops. Numeric drift from GPU floats is accepted and validated by output-count stability (instances/views within ~10% of the `run_c8a1fae3` baseline: 20 instances / 154 views) plus a visual spot-check of crops.

**Cannot be verified locally (no CUDA):** TRT export/speed, real NVDEC overlap, GPU-op speed. All are RunPod-only (Phase D). Correctness of the GPU math (DCT, scoring, glare) IS unit-tested on CPU tensors against the cv2 reference within tolerance.

**Baseline to beat (`run_c8a1fae3`):** total 31.6s; fused_inference 17.2s (YOLO 11.2s); refine CPU ops `quality_scoring` 2.16s + `glare_mask` 1.94s + `phash` 0.20s + `glare_centroid` 0.16s; VRAM 66%; 20 instances / 154 views.

---

## File Structure

| File | Phase | Change |
|---|---|---|
| `Dockerfile.cuda` | A | add `tensorrt` to the pip layer |
| `src/card_capture/detectors.py` | A | engine export/cache/load with FP16→FP32 `.pt` fallback in `_load_model`; pass `half=True` |
| `docker/start.sh` | A | build+cache the TRT engine during the warmup |
| `pipeline/steps/detect.py` | B | prefetch thread feeding the fused inference loop |
| `src/card_capture/gpu_refinement.py` | C | `warp_canonical_batch_gpu(..., return_gpu=True)` returns GPU tensors |
| `src/card_capture/ml/gpu_ops.py` | C | NEW: batched GPU primitives — `gpu_dct2`, `phash_batch`, `glare_mask_batch`, `glare_centroid_batch`, `spatial_glare_batch`, `laplacian_var_batch`, `rgb_gray_batch` |
| `src/card_capture/scoring.py` | C | `QualityScorer.score_batch(images_gpu, confidences, novelties)` |
| `pipeline/steps/refine.py` | C | per-view Python loop → one batched GPU scoring pass |
| `tests/...` | all | per task |

---

# PHASE A — TensorRT YOLO (FP16) with graceful fallback

## Task A1: engine export/cache/load in the detector

**Files:**
- Modify: `src/card_capture/detectors.py` (`_load_model`, ~lines 317-342; and `detect_batch`'s `model(...)` call ~line 256)
- Test: `tests/test_detector_trt.py`

- [ ] **Step 1: Write the failing test** (CPU-safe — mocks `YOLO` and the filesystem; verifies the fallback ladder and that an existing engine is preferred)

```python
# tests/test_detector_trt.py
"""Detector picks a cached .engine, else exports it, else falls back to .pt FP16."""
from unittest.mock import MagicMock, patch
import card_capture.detectors as det


def _make_detector(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "_resolve_model_path", lambda repo, fn: str(tmp_path / "model.pt"))
    (tmp_path / "model.pt").write_bytes(b"x")
    d = det.CardcaptorUltralyticsDetector(device="cuda")
    return d


def test_loads_existing_engine(tmp_path, monkeypatch):
    d = _make_detector(tmp_path, monkeypatch)
    (tmp_path / "model.engine").write_bytes(b"e")  # pretend a cached engine exists
    made = {}
    def _yolo(path):
        made["path"] = path
        m = MagicMock(); m.stride = 32; return m
    monkeypatch.setattr(det, "YOLO", _yolo, raising=False)
    monkeypatch.setattr(d, "_resolve_device", lambda: "cuda")
    d._load_model()
    assert made["path"].endswith("model.engine")  # preferred over .pt


def test_exports_engine_when_missing(tmp_path, monkeypatch):
    d = _make_detector(tmp_path, monkeypatch)
    calls = {"export": 0, "load": []}
    class _M:
        stride = 32
        def export(self, **kw):
            calls["export"] += 1
            assert kw["format"] == "engine" and kw["half"] is True
            (tmp_path / "model.engine").write_bytes(b"e")
            return str(tmp_path / "model.engine")
        def to(self, d): return self
    def _yolo(path):
        calls["load"].append(path)
        return _M()
    monkeypatch.setattr(det, "YOLO", _yolo, raising=False)
    monkeypatch.setattr(d, "_resolve_device", lambda: "cuda")
    d._load_model()
    assert calls["export"] == 1
    assert calls["load"][-1].endswith("model.engine")


def test_falls_back_to_pt_half_on_export_failure(tmp_path, monkeypatch):
    d = _make_detector(tmp_path, monkeypatch)
    class _M:
        stride = 32
        half_called = False
        def export(self, **kw):
            raise RuntimeError("no tensorrt")
        def to(self, d): return self
        def half(self): _M.half_called = True; return self
    monkeypatch.setattr(det, "YOLO", lambda p: _M(), raising=False)
    monkeypatch.setattr(d, "_resolve_device", lambda: "cuda")
    d._load_model()
    # On export failure we keep the .pt model (engine load not attempted with a bad path)
    assert d._model is not None
```

- [ ] **Step 2: Run → FAIL** (`_load_model` has no engine logic).
  Run: `source .venv/bin/activate 2>/dev/null; python3 -m pytest tests/test_detector_trt.py -q`

- [ ] **Step 3: Implement** — replace `_load_model` (current lines 317-342) with the engine-aware version:

```python
    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Real card detection requires optional dependencies. "
                "Install with: pip install '.[model]'"
            ) from exc

        self._device = self._resolve_device()
        model_path = _resolve_model_path(self.repo_id, self.filename)

        # TensorRT fast path (CUDA only). Engines are GPU/TRT-version specific and
        # cannot be prebuilt in the image, so we build-once-cache on the worker.
        # Ladder: cached .engine → export .engine → FP16 .pt → FP32 .pt.
        if self._device == "cuda":
            import os
            engine_path = os.path.splitext(model_path)[0] + ".engine"
            try:
                if not os.path.exists(engine_path):
                    exporter = YOLO(model_path)
                    # dynamic batch so the final short batch still runs; half=FP16.
                    out = exporter.export(format="engine", half=True, dynamic=True,
                                          imgsz=self.detection_width, device=0, verbose=False)
                    engine_path = str(out) if out else engine_path
                self._model = YOLO(engine_path)
                print(f"[detector] backend=tensorrt engine={engine_path}", flush=True)
                self._half = True
                return self._model
            except Exception as e:
                print(f"[detector] TensorRT unavailable ({e}); falling back to .pt FP16", flush=True)

        self._model = YOLO(model_path)
        self._model.to(self._device)
        # FP16 on CUDA even without TRT — the cheap ~2x win and our safety net.
        self._half = False
        if self._device == "cuda":
            try:
                self._model.half()
                self._half = True
            except Exception:
                pass
        print(f"[detector] backend={self._device} half={self._half}", flush=True)
        return self._model
```

Add `self._half = False` to `__init__` (near `self._model = None`, line ~204). In `detect_batch`, change the inference call (line 256) to pass half so the `.pt` FP16 path matches the engine's precision:

```python
        results = model(detect_images, conf=confidence_threshold, half=getattr(self, "_half", False), verbose=False)
```

(`model(...)` accepts both a TRT engine and a `.pt` model transparently; corners come back in the input-image space exactly as today, so `detect_batch`'s scale-back math is unchanged.)

- [ ] **Step 4: Run → PASS.** `python3 -m pytest tests/test_detector_trt.py -q`

- [ ] **Step 5: Confirm non-CUDA path unaffected** (MPS/CPU dev): `python3 -m pytest tests/ -q -k "detector or detect" 2>&1 | tail -8` — no new failures vs the pre-existing baseline.

- [ ] **Step 6: Commit**
```bash
git add src/card_capture/detectors.py tests/test_detector_trt.py
git commit -m "perf(detect): TensorRT FP16 engine with cached export + .pt fallback ladder

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

## Task A2: build the engine in the container warmup + add the dep

**Files:**
- Modify: `Dockerfile.cuda` (pip layer ~line 75-77)
- Modify: `docker/start.sh` (warmup `cc_warmup.py`, ~line 49-56)

- [ ] **Step 1: Add the dependency.** In `Dockerfile.cuda`, add `"tensorrt"` to the pip install list alongside `"ultralytics>=8.1.0"`:
```dockerfile
RUN pip install \
        "ultralytics>=8.1.0" \
        "tensorrt" \
        ...   # (keep the rest of the existing list unchanged)
```

- [ ] **Step 2: Build the engine during warmup.** In `docker/start.sh`'s `cc_warmup.py` heredoc, replace the YOLO warmup block (the `m = YOLO(weights); m.predict(...)` lines, ~53-56) with an export-then-predict that primes the cached engine on the real GPU:
```python
from ultralytics import YOLO
t = time.time()
import os
engine = os.path.splitext(weights)[0] + ".engine"
try:
    if not os.path.exists(engine):
        YOLO(weights).export(format="engine", half=True, dynamic=True, imgsz=640, device=0, verbose=False)
    m = YOLO(engine)
    print(f"trt engine ready: {(time.time()-t)*1000:.0f}ms", flush=True)
except Exception as e:
    print(f"trt export failed ({e}); warming .pt fp16", flush=True)
    m = YOLO(weights); m.half()
m.predict(np.zeros((640,640,3), dtype=np.uint8), device="cuda", imgsz=640, half=True, verbose=False)
print(f"yolo warmup: {(time.time()-t)*1000:.0f}ms", flush=True)
```

- [ ] **Step 3: Verify shell syntax** (no Python execution locally): `bash -n docker/start.sh && echo "start.sh OK"`. Expected: `start.sh OK`.

- [ ] **Step 4: Commit**
```bash
git add Dockerfile.cuda docker/start.sh
git commit -m "build(runpod): install tensorrt, build+cache YOLO engine in warmup

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

**Operational note (carry into Phase D):** the warmup’s first-ever export adds ~1–3 min to a cold start; warm workers reuse the cached `.engine`. The 120s warmup `timeout` in `start.sh` may need raising to ~240s to let the export finish — confirm the warmup log shows `trt engine ready`; if it times out, raise the timeout. The `dynamic=True` export must be re-confirmed to accept our variable last-batch size at runtime.

---

# PHASE B — decode/YOLO/warp prefetch overlap

## Task B1: prefetch thread in `_run_cuda_inference`

**Files:**
- Modify: `pipeline/steps/detect.py` (the `for gpu_batch, frames in sampler.sample_gpu_batches(...)` loop in `_run_cuda_inference`)
- Test: `tests/pipeline/test_detect_prefetch.py`

- [ ] **Step 1: Write the failing test** — verifies the loop still produces identical detection_rows/crops when batches arrive via a prefetch queue (order preserved), using the same mock shape as `test_detect_crop_cache.py`:

```python
# tests/pipeline/test_detect_prefetch.py
"""Prefetched inference yields the same detections, in frame order, as serial."""
import numpy as np, pytest
from unittest.mock import MagicMock
torch = pytest.importorskip("torch")


def test_prefetch_preserves_order_and_detections(tmp_path, monkeypatch):
    from pipeline.steps import detect
    from pipeline.steps.start import RunContext
    from card_capture.models import FrameSample, DetectionPacket, CornerDetection

    ctx = RunContext(video_path="/x.MOV", output_dir=str(tmp_path), db_path=str(tmp_path/"c.sqlite"),
                     detector="cuda", config_preset="balanced", crops_dir=str(tmp_path/"crops"),
                     frame_dir=str(tmp_path/"frames"), rotate_180=False, kornia_device="cpu", video_id=1)

    H, W = 64, 64
    def _frame(fi): return FrameSample(frame_index=fi, timestamp_ms=fi*16,
                                       image=np.zeros((H,W,3),dtype=np.uint8), width=W, height=H)
    batches = [(torch.zeros((2,H,W,3),dtype=torch.uint8), [_frame(0),_frame(1)]),
               (torch.zeros((2,H,W,3),dtype=torch.uint8), [_frame(2),_frame(3)])]
    sampler = MagicMock(); sampler.last_selected_frame_count=4; sampler.last_source_fps=30.0
    sampler.sample_gpu_batches.return_value = iter(batches)

    cd = lambda: CornerDetection(corners=[(0.,0.),(1.,0.),(1.,1.),(0.,1.)], confidence=0.9)
    detector = MagicMock(); detector.confidence_threshold=0.5; detector.detection_width=64
    detector.detect_batch.side_effect = lambda packets, conf: [
        DetectionPacket(frame_index=p.frame_index, timestamp_ms=p.timestamp_ms,
                        width=p.width, height=p.height, corner_detection=cd()) for p in packets]
    monkeypatch.setattr(detect, "KorniaNormalizer", lambda **k: MagicMock(
        warp_canonical_batch_gpu=lambda items, rotate_180=False: [np.zeros((1050,750,3),np.uint8) for _ in items]))

    crop_cache = {}
    out = detect._run_cuda_inference(ctx, sampler, detector, tmp_path, tmp_path, crop_cache=crop_cache)
    assert [r["frame_index"] for r in out.detection_rows] == [0,1,2,3]   # order preserved
    assert out.frame_count == 4
    assert len(crop_cache) == 4
```

- [ ] **Step 2: Run → FAIL or PASS-trivially.** (It may pass on the serial impl; that's fine — it's the regression guard for the refactor.) Run it and note the result.

- [ ] **Step 3: Implement the prefetch.** Wrap the sampler generator in a bounded-queue producer thread. Replace `for gpu_batch, frames in sampler.sample_gpu_batches(batch_size=batch_size, thumbnail_width=detector.detection_width):` with:

```python
        import queue as _queue
        import threading as _threading

        _q: "_queue.Queue" = _queue.Queue(maxsize=2)
        _SENTINEL = object()

        def _producer():
            try:
                for item in sampler.sample_gpu_batches(
                    batch_size=batch_size, thumbnail_width=detector.detection_width
                ):
                    _q.put(item)          # blocks while 2 batches are buffered (RAM/VRAM bound)
            except Exception as e:        # surface decode errors to the consumer
                _q.put(("__error__", e))
            finally:
                _q.put(_SENTINEL)

        _producer_thread = _threading.Thread(target=_producer, daemon=True)
        _producer_thread.start()

        while True:
            item = _q.get()
            if item is _SENTINEL:
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                raise item[1]
            gpu_batch, frames = item
            # ── unchanged loop body from here ──
            total_frames += len(frames)
            ...
```

Keep the entire existing loop body (packets, detect_batch, pos_by_frame, warp, crop_cache) verbatim under the `while`. The GIL is released during decord decode and torch CUDA calls, so the producer's decode of batch N+1 overlaps the consumer's YOLO+warp on batch N. `maxsize=2` bounds resident batches (≈2× the per-batch VRAM, which the 66% headroom accommodates).

- [ ] **Step 4: Run → PASS.** `python3 -m pytest tests/pipeline/test_detect_prefetch.py tests/pipeline/test_detect_crop_cache.py -q`

- [ ] **Step 5: Commit**
```bash
git add pipeline/steps/detect.py tests/pipeline/test_detect_prefetch.py
git commit -m "perf(detect): prefetch-thread decode so NVDEC overlaps YOLO+warp

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

# PHASE C — GPU-resident batched scoring

## Task C1: warp can return GPU-resident tensors

**Files:**
- Modify: `src/card_capture/gpu_refinement.py` (`warp_canonical_batch_gpu`, `_warp_from_stacked`)
- Test: `tests/test_gpu_warp_tensor.py` (add a case)

- [ ] **Step 1: Add a test** that `return_gpu=True` yields a torch tensor batch `(N,3,H,W)` or `(N,H,W,3)` on device, and that `.cpu()` of it equals the numpy return:
```python
def test_warp_gpu_return_matches_numpy(tmp_path):
    import torch, numpy as np
    from card_capture.gpu_refinement import KorniaNormalizer
    norm = KorniaNormalizer(width=750, height=1050, device="cpu")
    img = np.random.randint(0,256,(300,200,3),dtype=np.uint8)
    corners=[(10.,10.),(180.,12.),(185.,280.),(8.,275.)]
    cpu_list = norm.warp_canonical_batch_gpu([(torch.from_numpy(img),corners)], rotate_180=False)
    gpu = norm.warp_canonical_batch_gpu([(torch.from_numpy(img),corners)], rotate_180=False, return_gpu=True)
    assert hasattr(gpu, "shape") and gpu.shape[0] == 1            # batched tensor
    assert np.array_equal(cpu_list[0], gpu_to_bgr_hwc_numpy(gpu)[0])
```
where the plan provides `gpu_to_bgr_hwc_numpy` inline in the test as `lambda t: t.permute(0,2,3,1).cpu().numpy() if t.shape[1]==3 else t.cpu().numpy()` matching whatever layout C1 returns. **Pick ONE layout and document it**: return `(N, H, W, 3) uint8 BGR` on device (mirrors the numpy contract, simplest for downstream gray/threshold ops). Adjust the test accordingly.

- [ ] **Step 2: Run → FAIL** (no `return_gpu` kwarg).

- [ ] **Step 3: Implement.** Add `return_gpu: bool = False` to `warp_canonical_batch_gpu` and thread it into `_warp_from_stacked`. In `_warp_from_stacked`, when `return_gpu`, stop before `.cpu().numpy()` and instead return the device uint8 `(N,H,W,3)` BGR tensor (apply the `rotate_180` via `torch.rot90(x, 2, dims=[1,2])` on-device instead of `cv2.rotate`):
```python
    def _warp_from_stacked(self, batch_u8, matrices_np, rotate_180, return_gpu=False):
        batch_t = batch_u8.permute(0,3,1,2)[:, [2,1,0], :, :].float() / 255.0
        del batch_u8
        batch_m = torch.from_numpy(np.stack(matrices_np,0).astype(np.float32)).to(self.device, non_blocking=True)
        warped = kornia.geometry.transform.warp_perspective(batch_t, batch_m, (self.height, self.width))
        del batch_t
        out = (warped[:, [2,1,0], :, :] * 255.0).clamp_(0,255).to(torch.uint8).permute(0,2,3,1).contiguous()  # (N,H,W,3) BGR
        del warped
        if rotate_180:
            out = torch.rot90(out, 2, dims=[1,2])
        if return_gpu:
            return out                       # device tensor, stays resident
        np_out = out.cpu().numpy()
        return [np_out[i] for i in range(np_out.shape[0])]
```
`warp_canonical_batch_gpu(..., return_gpu)` passes the flag through; numpy `warp_canonical_batch` calls with `return_gpu=False` (unchanged behavior — verify the existing bit-identity tests still pass).

- [ ] **Step 4: Run → PASS.** `python3 -m pytest tests/test_gpu_warp_tensor.py -q`
- [ ] **Step 5: Commit** `perf(warp): optional GPU-resident tensor return for batched scoring`.

## Task C2: GPU primitives module — DCT pHash + glare + laplacian (batched)

**Files:**
- Create: `src/card_capture/ml/gpu_ops.py`
- Test: `tests/test_gpu_ops.py`

These run on CPU tensors in tests (torch CPU), validated against the cv2 reference within tolerance.

- [ ] **Step 1: Write the failing tests** (key correctness anchors):
```python
# tests/test_gpu_ops.py
import numpy as np, pytest
torch = pytest.importorskip("torch")
import cv2
from card_capture.ml import gpu_ops


def test_gpu_dct2_matches_cv2():
    x = np.random.rand(32,32).astype(np.float32)
    ref = cv2.dct(x)
    got = gpu_ops.gpu_dct2(torch.from_numpy(x)[None])[0].numpy()
    assert np.allclose(got, ref, atol=1e-3)


def test_phash_batch_matches_reference():
    # Two identical images → hamming distance 0; an inverted image → large distance.
    img = (np.random.rand(1050,750,3)*255).astype(np.uint8)
    batch = torch.from_numpy(np.stack([img, img, 255-img]))   # (3,H,W,3) BGR uint8
    hashes = gpu_ops.phash_batch(batch)                       # list[str] of 64 bits
    from card_capture.deduplicator import VisualDeduplicator
    d = VisualDeduplicator()
    assert d.hamming_distance(hashes[0], hashes[1]) == 0
    assert d.hamming_distance(hashes[0], hashes[2]) > 10


def test_glare_mask_batch_matches_cv2():
    img = (np.random.rand(20,20,3)*255).astype(np.uint8)
    batch = torch.from_numpy(img[None])                       # (1,H,W,3) BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref = (cv2.threshold(gray,200,255,cv2.THRESH_BINARY)[1]).astype(np.uint8)
    got = gpu_ops.glare_mask_batch(batch)[0].numpy().astype(np.uint8)*255
    assert np.array_equal(got, ref)


def test_laplacian_var_batch_close_to_cv2():
    img = (np.random.rand(100,100)*255).astype(np.uint8)
    ref = cv2.Laplacian(img, cv2.CV_64F).var()
    got = gpu_ops.laplacian_var_batch(torch.from_numpy(img)[None].float())[0].item()
    assert abs(got - ref) / max(ref,1.0) < 0.05               # within 5% (border handling differs)
```

- [ ] **Step 2: Run → FAIL** (module missing).

- [ ] **Step 3: Implement `src/card_capture/ml/gpu_ops.py`** — all batched, device-agnostic (work on whatever device the input tensor is on):
```python
"""Batched GPU image primitives for quality scoring. Inputs are torch tensors
(N,H,W,3) uint8 BGR on any device; ops stay on that device. cv2-equivalent
within float tolerance (numerics intentionally differ — see plan)."""
from __future__ import annotations
from typing import List
import torch
import torch.nn.functional as F

_DCT_CACHE: dict = {}

def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    key = (n, device, dtype)
    if key not in _DCT_CACHE:
        k = torch.arange(n, device=device, dtype=dtype)
        i = k.view(n, 1)
        D = torch.cos((2 * k.view(1, n) + 1) * i * torch.pi / (2 * n))
        D *= torch.sqrt(torch.tensor(2.0 / n, device=device, dtype=dtype))
        D[0] *= 1.0 / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
        _DCT_CACHE[key] = D
    return _DCT_CACHE[key]

def gpu_dct2(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D DCT-II of a batch (N,n,n), matching cv2.dct on n×n floats."""
    n = x.shape[-1]
    D = _dct_matrix(n, x.device, x.dtype)
    return D @ x @ D.transpose(-1, -2)

def rgb_gray_batch(bgr_u8: torch.Tensor) -> torch.Tensor:
    """(N,H,W,3) BGR uint8 → (N,H,W) float gray, matching cv2 BGR2GRAY weights."""
    b, g, r = bgr_u8[..., 0].float(), bgr_u8[..., 1].float(), bgr_u8[..., 2].float()
    return 0.114 * b + 0.587 * g + 0.299 * r

def phash_batch(bgr_u8: torch.Tensor) -> List[str]:
    """16-char hex perceptual hashes, byte-compatible with
    VisualDeduplicator.compute_phash / .hamming_distance (which does int(h,16)).
    Mirrors it exactly: 20% margin inner-crop → gray → 32×32 area-resize → DCT →
    8×8 low-freq vs median → 64 bits → hex."""
    n = bgr_u8.shape[0]
    h, w = bgr_u8.shape[1], bgr_u8.shape[2]
    # inner crop: drop 20% on each side (matches compute_phash: int(h*0.2))
    by, bx = int(h * 0.2), int(w * 0.2)
    inner = bgr_u8[:, by:h - by, bx:w - bx, :]
    gray = rgb_gray_batch(inner).unsqueeze(1)                       # (N,1,h',w')
    resized = F.interpolate(gray, size=(32, 32), mode="area").squeeze(1)
    dct = gpu_dct2(resized)
    low = dct[:, :8, :8].reshape(n, 64)
    med = low.median(dim=1, keepdim=True).values
    bits = (low > med)
    out = []
    for i in range(n):
        bitstr = "".join("1" if b else "0" for b in bits[i].tolist())
        out.append(f"{int(bitstr, 2):016x}")                       # 16-char hex, like compute_phash
    return out

def glare_mask_batch(bgr_u8: torch.Tensor) -> torch.Tensor:
    """(N,H,W,3)→(N,H,W) bool, gray>200 (cv2 THRESH_BINARY @200)."""
    return rgb_gray_batch(bgr_u8) > 200

def glare_centroid_batch(bgr_u8: torch.Tensor):
    """Returns list of (x,y) or None — centroid of gray>200 mask (== cv2.moments)."""
    mask = (rgb_gray_batch(bgr_u8) > 200).float()
    n, h, w = mask.shape
    ys = torch.arange(h, device=mask.device).view(1, h, 1).float()
    xs = torch.arange(w, device=mask.device).view(1, 1, w).float()
    area = mask.sum(dim=[1, 2])
    cx = (mask * xs).sum(dim=[1, 2]) / area.clamp(min=1)
    cy = (mask * ys).sum(dim=[1, 2]) / area.clamp(min=1)
    out = []
    for i in range(n):
        out.append((float(cx[i]), float(cy[i])) if area[i] > 0 else None)
    return out

def laplacian_var_batch(gray_f: torch.Tensor) -> torch.Tensor:
    """(N,H,W) float → (N,) variance of 3×3 Laplacian (kornel matches cv2's)."""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=gray_f.device, dtype=gray_f.dtype).view(1, 1, 3, 3)
    lap = F.conv2d(gray_f.unsqueeze(1), k, padding=1).squeeze(1)
    return lap.reshape(lap.shape[0], -1).var(dim=1, unbiased=False)

def spatial_glare_batch(bgr_u8: torch.Tensor) -> torch.Tensor:
    """GPU APPROXIMATION of scoring._spatial_glare_score.

    cv2 used connectedComponents to find the LARGEST saturated blob. No clean GPU
    CC exists, so we approximate "largest dense blob fraction" via average-pooling
    the saturation mask with a wide kernel: the max pooled density × kernel area
    estimates the biggest contiguous saturated region. Returns (N,) in [0,1],
    1=no glare (same convention as the cv2 score). Documented divergence."""
    # saturation ~ V channel high; approximate V with max(B,G,R)
    v = bgr_u8.float().amax(dim=-1)                                 # (N,H,W)
    sat = (v > 240).float().unsqueeze(1)                            # (N,1,H,W)
    n, _, h, w = sat.shape
    ksz = max(1, min(h, w) // 16)                                   # ~6% window
    dens = F.avg_pool2d(sat, kernel_size=ksz, stride=max(1, ksz // 2))
    blob_frac = (dens.amax(dim=[2, 3]).squeeze(1) * (ksz * ksz)) / float(h * w)
    return (1.0 - blob_frac * 10.0).clamp(0.0, 1.0)
```

- [ ] **Step 4: Run → PASS.** The 20% margin and 16-char-hex output are pinned to match `compute_phash`/`hamming_distance` exactly; if `gpu_dct2` tolerance trips, tune the DCT normalization (the pHash test asserts *relative* hamming, robust to small drift).
- [ ] **Step 5: Commit** `feat(gpu): batched GPU primitives — dct/phash/glare/laplacian`.

## Task C3: `QualityScorer.score_batch`

**Files:**
- Modify: `src/card_capture/scoring.py`
- Test: `tests/test_score_batch.py`

- [ ] **Step 1: Test** — `score_batch` over N GPU crops returns N `QualityScore`s; for a given image its `total` is within a tolerance of the existing per-image `score()` (drift accepted; assert closeness, not equality, and assert ordering is preserved for a sharp-vs-blurry pair):
```python
def test_score_batch_close_to_single_and_orders_sharpness(...):
    # build a sharp crop and a gaussian-blurred crop, both 1050x750 BGR
    # single = [scorer.score(sharp,0.9), scorer.score(blur,0.9)]
    # batch  = scorer.score_batch(torch.stack([sharp_t, blur_t]), [0.9,0.9], [1.0,1.0])
    # assert batch[0].total > batch[1].total                       # sharp ranks higher
    # assert abs(batch[0].components["sharpness"] - single[0].components["sharpness"]) < 0.1
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `score_batch(self, images: "torch.Tensor", confidences, novelties) -> List[QualityScore]`** in `QualityScorer`. `images` is `(N,H,W,3)` uint8 BGR on device (the warp output). Compute every component batched via `gpu_ops`, replicating `score()`'s formulas/weights exactly (aspect_ratio & size are constant since all crops are `width×height` — compute once; `complexity` weight is 0.0 — omit; `occlusion` neutral 1.0). Sharpness = `clamp(laplacian_var_batch(gray)/1000)`; glare = `clamp(1 - (gray>=245).mean()*4)`; border_purity = batched ring-vs-interior std; spatial_glare = `gpu_ops.spatial_glare_batch`. Move the final per-item `QualityScore` assembly into a short Python loop over the N precomputed component tensors (cheap — no per-image cv2). Keep the existing per-image `score()` for the non-CUDA path.

- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `feat(scoring): GPU-batched score_batch over warped crops`.

## Task C4: rewire `refine.py` to score the batch on GPU

**Files:**
- Modify: `pipeline/steps/refine.py` (the per-track / per-candidate scoring loop, ~lines 258-408)
- Modify: `pipeline/steps/detect.py` — fused path: warp with `return_gpu=True`, keep crops as GPU tensors in `crop_cache`
- Test: `tests/pipeline/test_refine_gpu_scoring.py`

- [ ] **Step 1: Test** — feed a fused-mode `refine.run` a `decoded_crops` of GPU tensors; assert it scores via `score_batch` (monkeypatch a spy), produces `frame_entries` with quality/glare/phash populated, writes crops to disk, and never calls the per-image `scorer.score`.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.** In `detect.py` fused warp, call `warp_canonical_batch_gpu(..., return_gpu=True)` and store the per-detection GPU tensor slices in `crop_cache` (keyed by detection_id, as today, but now device tensors). In `refine.py` fused mode (`decoded_crops is not None`): gather the chosen candidates’ crops into one batched tensor, call `scorer.score_batch`, `gpu_ops.glare_mask_batch`, `gpu_ops.glare_centroid_batch`, `gpu_ops.phash_batch` ONCE for all of them, then distribute results into `frame_entries`. Download each crop to numpy only at `cv2.imwrite` time (the masks for the fuse step are downloaded compressed as today). Remove the per-candidate `scorer.score`/`find_glare_centroid`/`compute_phash`/`_glare_mask` calls on the fused path. Non-fused path unchanged.

- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `perf(refine): one GPU-batched scoring pass over resident crops`.

---

# PHASE D — verification

## Task D1: regression + RunPod telemetry

- [ ] **Step 1: Full local suite vs branch point**, holding `card_capture_config.json` constant (the runpod-backend value makes one app integration test fail regardless — see prior runs). Method: capture `FAILED` set on HEAD and on the merge-base with the same config, `comm` them. Expected: zero NEW failures. Run:
  `python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py`
- [ ] **Step 2: Deploy** (push to `origin/main`) and confirm in the next `run_*_handler_output.json`:
  - `detect_telemetry.yolo_elapsed_s`: 11.2s → ~3–4s (engine) or ~5–6s (FP16 fallback). Check the worker log for `backend=tensorrt` vs `half=True`.
  - refine `op_seconds`: `quality_scoring`/`glare_mask`/`phash`/`glare_centroid` → near-0 (batched on GPU).
  - `fused_inference_s` drop reflects decode hidden behind YOLO (Phase B): compare to the serial sum.
  - `gpu_pct` mean up; total stage time toward single digits.
  - **Output stability:** `card_instances_this_run` and `card_views_total` within ~10% of 20 / 154. Spot-check `crops/` — confirm pHash/scoring drift didn't change which frames were selected in a visually-wrong way.
- [ ] **Step 3:** If TRT cold-start export overruns the `start.sh` 120s warmup `timeout`, raise it (Task A2 note) and redeploy.

---

## Self-Review

**Spec coverage:** #1 TensorRT → Phase A (A1 load ladder + A2 build/dep/warmup). #2 full-GPU scoring → Phase C (C1 resident crops, C2 GPU primitives incl. approximated spatial_glare + DCT pHash, C3 score_batch, C4 refine rewire). #3 overlap → Phase B. ✅

**Placeholder scan:** concrete code for the load ladder, prefetch, all GPU primitives, and the warp return; C3/C4 give exact component mapping + signatures and reference the C2 primitives by name. Test code present per task. ✅

**Type consistency:** `warp_canonical_batch_gpu(..., return_gpu=True)` → `(N,H,W,3)` uint8 BGR device tensor; `crop_cache` holds those; `score_batch(images:(N,H,W,3) uint8, ...)`, `glare_mask_batch`, `phash_batch`, `glare_centroid_batch` all consume that same layout (C2 docstrings fix it). `gpu_dct2` consumes `(N,n,n)`. ✅

**Risks:** (1) TRT engine cold-start build time + ephemeral workers — mitigated by warmup build + FP16/FP32 fallback ladder (worst case = today’s behavior). (2) `spatial_glare_batch` and pHash are approximations — weight on spatial_glare is only 0.05; pHash validated by relative hamming; both gated by the output-count check in D1. (3) GPU float vs cv2 float64 score drift can shift canonical/prune decisions — explicitly validated against the 20/154 baseline + crop spot-check. (4) Phase B `maxsize=2` raises peak VRAM ~1 batch — fits in the 66%→~ headroom but confirm no OOM in D1/Phase D.
