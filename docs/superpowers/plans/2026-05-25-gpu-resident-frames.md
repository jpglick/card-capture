# GPU-Resident Frames (decode→YOLO→warp without full-frame CPU round-trips) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop downloading every full 4K frame from GPU to CPU and re-uploading it for the warp. Keep the decoded 4K frame resident on the GPU; only ever move *small* data across PCIe (a 640px YOLO thumbnail + the final 750×1050 crops).

**Architecture:** decord (with the torch bridge) already decodes to a GPU tensor; today `CudaSampler` immediately does `.cpu().numpy()` on it (a ~25 GB download), and the warp later re-uploads the full frame (~25 GB). This plan adds a GPU-resident batch generator (`CudaSampler.sample_gpu_batches`) that keeps the 4K tensor on the GPU and produces a small CPU thumbnail for YOLO via a GPU resize, plus a GPU-tensor warp entry point (`KorniaNormalizer.warp_canonical_batch_gpu`) that warps directly from the resident tensor. The fused CUDA inference (`_run_cuda_inference`) is rewired to use both. YOLO's detector and ultralytics' preprocessing are **left untouched** — the detector still receives a numpy thumbnail and does its own letterbox/normalize/coord-mapping. Result: ~50 GB of round-trip PCIe traffic drops to ~2 GB.

**Tech Stack:** Python 3, decord (NVDEC + torch bridge), torch / `torch.nn.functional.interpolate` (GPU resize), Kornia (`warp_perspective`), Ultralytics YOLOv8-OBB, OpenCV, pytest.

**Scope decision (locked with user):** lower-risk path. YOLO keeps consuming a CPU numpy 640px thumbnail through ultralytics' unchanged numpy preprocessing. We do **not** feed a GPU tensor into ultralytics (that "aggressive" variant saves only ~0.7 GB more but requires reimplementing ultralytics' letterbox/normalize/coord-mapping and is RunPod-only verifiable).

**Channel-order safety:** `batch_data` (the decord GPU tensor) and today's `frames_np = batch_data.cpu().numpy()` are the *same bytes*. So a thumbnail resized from `batch_data` has the same channel order the detector sees today, and warping from a `batch_data` slice produces the same result as warping from `frames_np[i]` today. No color-space change is introduced anywhere.

**Why most of this is unit-testable without a GPU:** torch ops and `kornia.warp_perspective` run on CPU. The GPU-tensor warp and the resize are validated on CPU tensors against the existing numpy path. Only true NVDEC throughput/VRAM must be confirmed on RunPod (Task 4).

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/card_capture/gpu_refinement.py` | Modify | Refactor the warp into a shared stacked-tensor core; add `warp_canonical_batch_gpu` that accepts GPU uint8 HWC tensors (no upload). |
| `src/card_capture/sampler/cuda_sampler.py` | Modify | Extract loader setup into `_prepare_loader`; add `sample_gpu_batches` yielding `(gpu_tensor_batch, [FrameSample with thumbnail])`; add `_gpu_thumbnails` GPU-resize helper. |
| `pipeline/steps/detect.py` | Modify | Rewire `_run_cuda_inference` to use `sample_gpu_batches` (thumbnail → YOLO) and `warp_canonical_batch_gpu` (GPU tensor → warp). |
| `tests/test_gpu_warp_tensor.py` | Create | `warp_canonical_batch_gpu` matches `warp_canonical_batch` on identical data (CPU device). |
| `tests/test_cuda_sampler_gpu.py` | Create | `sample_gpu_batches` keeps the tensor and emits correctly-sized thumbnails (mocked decord). |
| `tests/pipeline/test_detect_crop_cache.py` | Modify | Update the existing crop-cache test to the GPU-resident loop. |

---

## Task 1: GPU-tensor warp entry point

**Files:**
- Modify: `src/card_capture/gpu_refinement.py:25-89` (`warp_canonical_batch`)
- Test: `tests/test_gpu_warp_tensor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_warp_tensor.py
"""warp_canonical_batch_gpu (GPU-tensor input) matches the numpy path bit-for-bit on CPU."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")

from card_capture.gpu_refinement import KorniaNormalizer


def _img_and_corners():
    img = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)  # H,W,3 BGR
    corners = [(10.0, 10.0), (180.0, 12.0), (185.0, 280.0), (8.0, 275.0)]
    return img, corners


def test_gpu_tensor_warp_matches_numpy():
    norm = KorniaNormalizer(width=750, height=1050, device="cpu")
    img, corners = _img_and_corners()

    from_numpy = norm.warp_canonical_batch([(img, corners)], rotate_180=False)
    from_tensor = norm.warp_canonical_batch_gpu(
        [(torch.from_numpy(img), corners)], rotate_180=False
    )

    assert len(from_numpy) == 1 and len(from_tensor) == 1
    assert from_numpy[0].shape == (1050, 750, 3)
    # Identical input data through the same warp core → identical output.
    assert np.array_equal(from_numpy[0], from_tensor[0])


def test_gpu_tensor_warp_empty_returns_empty():
    norm = KorniaNormalizer(width=750, height=1050, device="cpu")
    assert norm.warp_canonical_batch_gpu([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_gpu_warp_tensor.py -q`
Expected: FAIL — `KorniaNormalizer` has no attribute `warp_canonical_batch_gpu`.

- [ ] **Step 3: Refactor into a shared core and add the GPU entry**

In `src/card_capture/gpu_refinement.py`, replace the body of `warp_canonical_batch` (lines 25-89) with these four methods (shared `_perspective_matrix` + `_warp_from_stacked`, the existing numpy entry, and the new GPU entry):

```python
    def _perspective_matrix(self, corners: List[Point]) -> np.ndarray:
        """CPU-side 3x3 perspective matrix mapping the card quad → portrait canvas."""
        pts_dst = np.array(
            [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]],
            dtype=np.float32,
        )
        ordered = order_points_clockwise(corners)
        oriented = _orient_for_target_canvas(ordered, self.width, self.height)
        pts_src = np.array(oriented, dtype=np.float32)
        return cv2.getPerspectiveTransform(pts_src, pts_dst)

    def _warp_from_stacked(
        self, batch_u8, matrices_np: List[np.ndarray], rotate_180: bool
    ) -> List[np.ndarray]:
        """Warp a stacked uint8 (B,H,W,3) BGR tensor (already on device) → list of BGR crops.

        This is the single GPU warp core shared by the numpy and GPU-tensor entry
        points. Input/output channel handling is unchanged from the original
        warp_canonical_batch: BGR in (index-swapped to RGB for kornia, swapped
        back to BGR for cv2.imwrite consumers).
        """
        batch_t = batch_u8.permute(0, 3, 1, 2)[:, [2, 1, 0], :, :].float() / 255.0
        del batch_u8

        batch_m = torch.from_numpy(np.stack(matrices_np, axis=0)).to(
            self.device, non_blocking=True
        )
        warped = kornia.geometry.transform.warp_perspective(
            batch_t, batch_m, (self.height, self.width)
        )
        del batch_t

        warped_u8 = (warped[:, [2, 1, 0], :, :] * 255.0).clamp_(0, 255).to(torch.uint8)
        warped_u8 = warped_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        del warped

        images: List[np.ndarray] = []
        for bgr in warped_u8:
            if rotate_180:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            images.append(bgr)
        return images

    def warp_canonical_batch(
        self, batch_data: List[Tuple[Union[str, np.ndarray], List[Point]]], rotate_180: bool = True
    ) -> List[np.ndarray]:
        """Warp from numpy images (or image paths). Uploads to GPU, then warps.

        Optimized 2026-05-24 — uploads uint8 and does float/scale/channel reorder
        on the device (see _warp_from_stacked).
        """
        imgs: List[np.ndarray] = []
        mats: List[np.ndarray] = []
        for image_or_path, corners in batch_data:
            img = image_or_path if isinstance(image_or_path, np.ndarray) else cv2.imread(image_or_path)
            if img is None:
                continue
            imgs.append(img)
            mats.append(self._perspective_matrix(corners))
        if not imgs:
            return []
        batch_u8 = torch.from_numpy(np.stack(imgs, axis=0)).to(self.device, non_blocking=True)
        return self._warp_from_stacked(batch_u8, mats, rotate_180)

    def warp_canonical_batch_gpu(
        self, batch_data: List[Tuple["torch.Tensor", List[Point]]], rotate_180: bool = True
    ) -> List[np.ndarray]:
        """Warp from GPU-resident uint8 (H,W,3) BGR tensors — no host→device upload.

        Each item's image is a torch tensor already on the GPU (a slice of the
        decoded decord batch). Skips the np.stack/from_numpy/.to() upload that
        warp_canonical_batch pays.
        """
        tensors: List["torch.Tensor"] = []
        mats: List[np.ndarray] = []
        for img_t, corners in batch_data:
            if img_t is None:
                continue
            tensors.append(img_t)
            mats.append(self._perspective_matrix(corners))
        if not tensors:
            return []
        batch_u8 = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
        return self._warp_from_stacked(batch_u8, mats, rotate_180)
```

(`torch`, `kornia`, `cv2`, `np`, `List`, `Tuple`, `Union`, `Point`, `order_points_clockwise`, `_orient_for_target_canvas` are all already imported at the top of the file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_gpu_warp_tensor.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm the existing warp consumers are unaffected**

Run: `python3 -m pytest tests/pipeline/test_refine_fused.py tests/pipeline/test_refine_telemetry.py -q`
Expected: PASS (numpy `warp_canonical_batch` behavior is byte-identical — same ops, just factored).

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/gpu_refinement.py tests/test_gpu_warp_tensor.py
git commit -m "feat(warp): add GPU-tensor warp entry to skip host->device upload

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: GPU-resident sampler batches

**Files:**
- Modify: `src/card_capture/sampler/cuda_sampler.py` (extract `_prepare_loader` from `sample_batches`; add `_gpu_thumbnails` and `sample_gpu_batches`)
- Test: `tests/test_cuda_sampler_gpu.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cuda_sampler_gpu.py
"""sample_gpu_batches keeps the 4K tensor on-device and emits small thumbnails."""
import numpy as np
import pytest
from unittest.mock import MagicMock

torch = pytest.importorskip("torch")


def test_sample_gpu_batches_keeps_tensor_and_resizes_thumbnail(monkeypatch):
    import card_capture.sampler.cuda_sampler as cs

    # Fake a 2-frame batch of 3840-wide (portrait W=2160? use W=1920 for simple math) frames.
    H, W = 1080, 1920
    batch_tensor = torch.randint(0, 256, (2, H, W, 3), dtype=torch.uint8)
    indices = torch.tensor([[0, 4], [0, 6]])  # decord (video_idx, frame_idx) pairs

    fake_vr = MagicMock()
    fake_vr.__len__ = lambda self: 100
    fake_vr.get_avg_fps.return_value = 30.0
    fake_vr.__getitem__ = lambda self, i: torch.zeros((H, W, 3), dtype=torch.uint8)

    fake_loader = [(batch_tensor, indices)]

    fake_decord = MagicMock()
    fake_decord.cpu.return_value = "cpu_ctx"
    fake_decord.VideoReader.return_value = fake_vr
    fake_decord.VideoLoader.return_value = fake_loader
    monkeypatch.setattr(cs, "decord", fake_decord)

    sampler = cs.CudaSampler.__new__(cs.CudaSampler)
    sampler.video_path = "/fake/video.MOV"
    sampler.stride = 2
    sampler._gpu_ctx = "gpu_ctx"

    out = list(sampler.sample_gpu_batches(batch_size=2, thumbnail_width=640, video_path="/fake/video.MOV"))
    assert len(out) == 1
    gpu_batch, frames = out[0]

    # The full-res tensor is preserved (NOT downloaded to numpy).
    assert isinstance(gpu_batch, torch.Tensor)
    assert gpu_batch.shape == (2, H, W, 3)

    # Each FrameSample carries a 640-wide thumbnail but the ORIGINAL frame dims.
    assert len(frames) == 2
    for f in frames:
        assert f.width == W and f.height == H            # original dims for coord scale-back
        assert f.image.shape[1] == 640                   # thumbnail width
        assert f.image.shape[0] == round(H * 640 / W)    # aspect-preserved height
    assert [f.frame_index for f in frames] == [4, 6]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cuda_sampler_gpu.py -q`
Expected: FAIL — `CudaSampler` has no attribute `sample_gpu_batches`.

- [ ] **Step 3: Extract `_prepare_loader`, add `_gpu_thumbnails` and `sample_gpu_batches`**

In `src/card_capture/sampler/cuda_sampler.py`, add a module-level helper near the top (after the imports, before the class):

```python
def _gpu_thumbnails(batch_u8, target_w: int):
    """Resize a (N,H,W,3) uint8 tensor to width target_w on its device; return numpy.

    Aspect-preserving. The only data that crosses PCIe is the small thumbnail.
    """
    import torch
    import torch.nn.functional as F

    n, H, W, _ = batch_u8.shape
    if W <= target_w:
        return batch_u8.cpu().numpy()
    th = max(1, round(H * target_w / W))
    t = batch_u8.permute(0, 3, 1, 2).float()
    r = F.interpolate(t, size=(th, target_w), mode="bilinear", align_corners=False)
    return r.clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
```

Refactor the loader setup out of `sample_batches` into a shared method. Add to the `CudaSampler` class:

```python
    def _prepare_loader(self, batch_size: int, video_path):
        """Probe dims and build the decord VideoLoader. Returns (vl, fps, h, w) or (None, fps, h, w)."""
        resolved = Path(video_path) if video_path else self.video_path
        if resolved is None:
            raise ValueError("video_path must be provided")

        probe = decord.VideoReader(str(resolved), ctx=decord.cpu(0))
        total = len(probe)
        fps = probe.get_avg_fps() or 30.0
        first = probe[0].cpu().numpy()
        h, w = first.shape[:2]
        self.last_source_fps = fps
        self.last_selected_frame_count = max(1, (total + self.stride - 1) // self.stride)
        del probe

        if total == 0:
            return None, fps, h, w

        vl = decord.VideoLoader(
            [str(resolved)],
            ctx=[self._gpu_ctx],
            shape=(batch_size, h, w, 3),
            interval=max(0, self.stride - 1),
            skip=0,
            shuffle=0,
        )
        return vl, fps, h, w

    @staticmethod
    def _flatten_indices(batch_indices):
        """decord yields (N,2) [video_idx, frame_idx] pairs; take the frame_idx column."""
        idx_arr = batch_indices.cpu().numpy()
        if idx_arr.ndim == 2:
            return idx_arr[:, 1].astype(int)
        return idx_arr.reshape(-1).astype(int)

    def sample_gpu_batches(
        self,
        batch_size: int = 32,
        thumbnail_width: int = 640,
        video_path: Optional[Union[Path, str]] = None,
    ):
        """Yield (gpu_tensor_batch, [FrameSample]) keeping full-res frames on the GPU.

        The 4K tensor stays on-device for the warp; each FrameSample.image is a
        small CPU thumbnail (width=thumbnail_width) for YOLO, while width/height
        carry the ORIGINAL frame dims so the detector scales corners back to
        full-frame coordinates.
        """
        vl, fps, h, w = self._prepare_loader(batch_size, video_path)
        if vl is None:
            return
        for batch_data, batch_indices in vl:
            indices_flat = self._flatten_indices(batch_indices)
            thumbs = _gpu_thumbnails(batch_data, thumbnail_width)  # numpy (N, th, tw, 3)
            frames = [
                FrameSample(
                    frame_index=int(idx),
                    timestamp_ms=int(idx * 1000 / fps),
                    image=thumbs[i],
                    width=w,
                    height=h,
                )
                for i, idx in enumerate(indices_flat)
            ]
            yield batch_data, frames
```

Then update the existing `sample_batches` to reuse `_prepare_loader` and `_flatten_indices` (preserving its current numpy behavior). Replace its setup + loop body with:

```python
    def sample_batches(
        self,
        batch_size: int = 32,
        video_path: Optional[Union[Path, str]] = None,
    ) -> Iterator[list]:
        """Yield lists of FrameSample (numpy images) via VideoLoader. Back-compat path."""
        vl, fps, h, w = self._prepare_loader(batch_size, video_path)
        if vl is None:
            return
        for batch_data, batch_indices in vl:
            frames_np = batch_data.cpu().numpy()  # (N, H, W, 3)
            indices_flat = self._flatten_indices(batch_indices)
            batch = [
                FrameSample(
                    frame_index=int(idx),
                    timestamp_ms=int(idx * 1000 / fps),
                    image=frames_np[i],
                    width=w,
                    height=h,
                )
                for i, idx in enumerate(indices_flat)
            ]
            yield batch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cuda_sampler_gpu.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm `sample_batches` back-compat is intact**

Run: `python3 -m pytest tests/ -q -k "cuda_sampler or sampler" 2>&1 | tail -15`
Expected: no *new* failures versus the documented pre-existing `test_sampler.py` baseline (the `sample_batches` refactor is behavior-preserving).

- [ ] **Step 6: Commit**

```bash
git add src/card_capture/sampler/cuda_sampler.py tests/test_cuda_sampler_gpu.py
git commit -m "feat(sampler): add GPU-resident batch generator with thumbnail downscale

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: rewire `_run_cuda_inference` to the GPU-resident path

**Files:**
- Modify: `pipeline/steps/detect.py:200-296` (`_run_cuda_inference`)
- Test: `tests/pipeline/test_detect_crop_cache.py` (update)

- [ ] **Step 1: Update the existing test to the GPU-resident loop**

Replace the body of `tests/pipeline/test_detect_crop_cache.py` with:

```python
# tests/pipeline/test_detect_crop_cache.py
"""_run_cuda_inference warps from the GPU-resident tensor and fills the crop cache."""
import numpy as np
import pytest
from unittest.mock import MagicMock

torch = pytest.importorskip("torch")


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


def test_run_cuda_inference_warps_from_gpu_tensor(tmp_path, monkeypatch):
    from pipeline.steps import detect
    from card_capture.models import FrameSample, CornerDetection

    H, W = 2160, 3840
    gpu_batch = torch.zeros((1, H, W, 3), dtype=torch.uint8)
    thumb = np.zeros((360, 640, 3), dtype=np.uint8)
    frames = [FrameSample(frame_index=5, timestamp_ms=166, image=thumb, width=W, height=H)]

    sampler = MagicMock()
    sampler.last_selected_frame_count = 1
    sampler.last_source_fps = 30.0
    sampler.sample_gpu_batches.return_value = iter([(gpu_batch, frames)])

    corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 140.0), (0.0, 140.0)]
    detector = MagicMock()
    detector.confidence_threshold = 0.5
    detector.detection_width = 640

    def _detect_batch(packets, conf):
        out = []
        for p in packets:
            p.corner_detection = CornerDetection(corners=corners, confidence=0.9)
            out.append(p)
        return out
    detector.detect_batch.side_effect = _detect_batch

    captured = {}

    class _FakeNorm:
        def warp_canonical_batch_gpu(self, batch_items, rotate_180=False):
            captured["items"] = batch_items
            return [np.full((1050, 750, 3), i + 1, dtype=np.uint8) for i in range(len(batch_items))]
    monkeypatch.setattr(detect, "KorniaNormalizer", lambda **k: _FakeNorm())

    ctx = _make_ctx(tmp_path)
    crop_cache: dict = {}
    out = detect._run_cuda_inference(ctx, sampler, detector, tmp_path, tmp_path, crop_cache=crop_cache)

    assert len(out.detection_rows) == 1
    det_id = out.detection_rows[0]["detection_id"]
    assert det_id in crop_cache
    assert crop_cache[det_id].shape == (1050, 750, 3)
    # The warp received a GPU-tensor slice (not a numpy frame).
    assert isinstance(captured["items"][0][0], torch.Tensor)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/pipeline/test_detect_crop_cache.py -q`
Expected: FAIL — `_run_cuda_inference` still calls `sampler.sample_batches` and `warp_canonical_batch` (no `sample_gpu_batches` on the mock, and it doesn't pass tensor slices to the warp).

- [ ] **Step 3: Rewire the inference loop**

In `pipeline/steps/detect.py`, replace the body of `_run_cuda_inference` from the `for batch in sampler.sample_batches(...)` loop through the end of that loop (current lines 226 onward, up to the `return DetectOutput(...)`) with the GPU-resident loop. The normalizer-build block (added previously) stays. New loop:

```python
    for gpu_batch, frames in sampler.sample_gpu_batches(
        batch_size=batch_size, thumbnail_width=detector.detection_width
    ):
        total_frames += len(frames)

        packets_in = [
            FramePacket(
                frame_index=f.frame_index,
                timestamp_ms=f.timestamp_ms,
                image=f.image,            # 640px thumbnail; detector scales corners via width/height
                width=f.width,
                height=f.height,
                triage_metrics={},
            )
            for f in frames
        ]

        _t = _time.time()
        packets_out = detector.detect_batch(packets_in, detector.confidence_threshold)
        yolo_elapsed_s += _time.time() - _t
        yolo_batches += 1

        for f in frames:
            accepted_frame_presence.append((f.frame_index, f.timestamp_ms, True))

        # Position of each frame within this GPU batch, so we can slice the
        # resident 4K tensor for warping (no re-decode, no host upload).
        pos_by_frame = {f.frame_index: i for i, f in enumerate(frames)}

        warp_items = []   # (gpu_tensor_slice, corners)
        warp_ids = []
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
                pos = pos_by_frame.get(pkt.frame_index)
                if pos is not None:
                    warp_items.append((gpu_batch[pos], [(float(p[0]), float(p[1])) for p in cd.corners]))
                    warp_ids.append(this_id)
            det_id += 1

        if crop_cache is not None and warp_items:
            try:
                warped = normalizer.warp_canonical_batch_gpu(warp_items, rotate_180=ctx.rotate_180)
                if len(warped) != len(warp_ids):
                    raise RuntimeError(
                        f"warp count mismatch: {len(warped)} crops for {len(warp_ids)} detections"
                    )
                for wid, img in zip(warp_ids, warped):
                    crop_cache[wid] = img
            except Exception as e:
                print(f"[detect] crop warp failed for batch: {e}", flush=True)
```

(`FramePacket` is already imported at the top of `_run_cuda_inference`; `KorniaNormalizer` and `_time` are already imported. The normalizer-build block from the prior change — `normalizer = KorniaNormalizer(...)` guarded by `crop_cache is not None` — remains directly above this loop.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/pipeline/test_detect_crop_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Run the fused-path test suite**

Run: `python3 -m pytest tests/pipeline/test_fused_refine.py tests/pipeline/test_refine_fused.py tests/pipeline/test_flow_fused_branch.py -q`
Expected: PASS (`fused_refine` calls `_run_cuda_inference` unchanged at the seam; only its internals changed).

- [ ] **Step 6: Commit**

```bash
git add pipeline/steps/detect.py tests/pipeline/test_detect_crop_cache.py
git commit -m "perf(detect): warp from GPU-resident frames, only thumbnails cross PCIe

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: full regression + RunPod verification

**Files:** none (verification only)

- [ ] **Step 1: Full unit suite vs. the pre-existing baseline**

Run: `python3 -m pytest tests/ -q --ignore=tests/pipeline/test_path_equivalence.py 2>&1 | tail -25`
Expected: no *new* failures beyond the documented pre-existing set. Diff the `FAILED` list against the parent commit (`git stash`, checkout parent, run, compare) if anything looks unfamiliar — same method used for the fused-decode change.

- [ ] **Step 2: Deploy + RunPod telemetry check (manual, after merge to origin/main)**

`docker/start.sh` syncs workers to `origin/main` on container start, so pushing to main is the deploy. After a RunPod run completes, pull the newest `card_capture_output/run_*/run_*_handler_output.json` and confirm:
- **Transfer down:** `resource_stats.mem_io_pct` mean/peak during `detect` drops materially from the GPU-resident baseline (`run_d79a488b`: mem_io mean 13.2 / peak 78).
- **Decode-path time down:** `stage_payloads.stage_detect.fused_inference_s` drops well below ~36–40 s (the ~25 s of full-frame GPU↔CPU transfer should largely disappear; expect the pass to move toward the YOLO+warp+true-NVDEC sum).
- **GPU/decoder util up:** `resource_stats.decoder_pct` / `gpu_pct` means rise versus the ~11% / ~22% baseline (we stop stalling on PCIe).
- **Output preserved:** `card_instances_this_run` == 18 and `card_views_total` within ~10% of 139–140. Channel order unchanged → crops must look correct (spot-check a few in `crops/`).

---

## Self-Review

**Spec coverage:**
- Keep the 4K frame on the GPU (no full-frame `.cpu().numpy()`) → Task 2 (`sample_gpu_batches` keeps `batch_data`; only `_gpu_thumbnails` downloads a small image). ✅
- Warp from the resident tensor (no re-upload) → Task 1 (`warp_canonical_batch_gpu`) + Task 3 (passes `gpu_batch[pos]` slices). ✅
- YOLO/ultralytics untouched (lower-risk) → Task 3 feeds a numpy thumbnail to the unchanged `detect_batch`/ultralytics numpy path. ✅
- Only small data crosses PCIe (thumbnail + final crops) → Tasks 2+3. ✅

**Placeholder scan:** No TBD/TODO; every code step has complete code and exact commands. ✅

**Type consistency:**
- `sample_gpu_batches` yields `(torch.Tensor, list[FrameSample])`; Task 3 consumes exactly that (`for gpu_batch, frames in ...`). ✅
- `warp_canonical_batch_gpu(batch_data: list[(torch.Tensor, corners)])` defined in Task 1; called in Task 3 with `(gpu_batch[pos], corners)` items. ✅
- `_warp_from_stacked` consumes a stacked uint8 (B,H,W,3) device tensor from both entry points; returns `list[np.ndarray]` (B, 1050, 750, 3) BGR — same return type `refine`/`crop_cache` already expect. ✅
- `FrameSample.image` carries the thumbnail while `width`/`height` stay original — relied on by `detect_batch`'s `scale_x = frame.width / scaled_w` math (Task 3 rationale). ✅

**Risk notes:**
- Channel order is preserved by construction (thumbnail and warp both derive from the same `batch_data` bytes that `frames_np` used). Still explicitly spot-checked in Task 4 Step 2.
- VRAM is unchanged-to-slightly-higher (the resident 4K batch coexists with the warp working set, same as today's upload did). This change does not reduce the warp's float32 peak — that's the separate bbox-crop work. Confirm no OOM in Task 4; if tight, bbox-crop warp is the follow-up.
- True NVDEC throughput and the transfer win are GPU-only and verified on RunPod (Task 4 Step 2); local tests cover correctness via CPU tensors only.
