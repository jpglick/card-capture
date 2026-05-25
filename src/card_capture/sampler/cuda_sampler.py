"""NVDEC-accelerated stride sampler for the CUDA pipeline.

Uses decord.VideoLoader for continuous batched GPU decode — only batch_size
frames in RAM at once regardless of video length.

GPU-or-die: raises RuntimeError if NVDEC is unavailable and
CC_CUDA_ALLOW_CPU_FALLBACK is not set. Production containers never set that flag.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np

try:
    import decord
    # Use torch as the array bridge — see pipeline_utils.py for the full
    # rationale. Short version: shares torch's CUDA context (avoids deadlocks)
    # and returns torch.Tensor instead of decord.NDArray.
    decord.bridge.set_bridge('torch')
    _decord_import_error: str | None = None
except Exception as _e:
    decord = None  # type: ignore[assignment]
    _decord_import_error = f"{type(_e).__name__}: {_e}"

from card_capture.models import FrameSample


def _gpu_thumbnails(batch_u8, target_w: int):
    """Resize a (N,H,W,3) uint8 tensor to width target_w on its device; return numpy.

    Aspect-preserving. The only data that crosses PCIe is the small thumbnail.
    """
    import torch
    import torch.nn.functional as F

    n, H, W, _ = batch_u8.shape
    if W <= target_w:
        return batch_u8.contiguous().cpu().numpy()
    th = max(1, round(H * target_w / W))
    t = batch_u8.permute(0, 3, 1, 2).float()
    r = F.interpolate(t, size=(th, target_w), mode="bilinear", align_corners=False)
    return r.clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _probe_gpu() -> object:
    """Return a decord GPU context (index 0), or raise on failure."""
    return decord.gpu(0)


class CudaSampler:
    """Uniform-stride video sampler using decord VideoLoader for GPU decode.

    Args:
        video_path: Source video file.
        stride: Sample every Nth source frame. Default 2 = every other frame.
        opening_scan_s: Retained for API compatibility; no longer used.
    """

    def __init__(
        self,
        video_path: Optional[Union[Path, str]] = None,
        stride: int = 2,
        opening_scan_s: float = 2.0,
    ) -> None:
        self.video_path = Path(video_path) if video_path else None
        self.stride = max(1, stride)
        self.last_source_fps: float = 30.0
        self.last_selected_frame_count: int = 0

        allow_fallback = os.environ.get("CC_CUDA_ALLOW_CPU_FALLBACK", "0") == "1"
        if decord is None:
            if not allow_fallback:
                raise RuntimeError(
                    "CudaSampler requires decord with NVDEC. "
                    f"import decord failed: {_decord_import_error}. "
                    "Set CC_CUDA_ALLOW_CPU_FALLBACK=1 to allow CPU fallback."
                )
            raise RuntimeError(
                "CudaSampler requires decord; CPU fallback path needs decord too."
            )
        try:
            self._gpu_ctx = _probe_gpu()
        except Exception as e:
            if not allow_fallback:
                raise RuntimeError(
                    "CudaSampler requires NVDEC (decord GPU context). "
                    f"decord.gpu(0) raised: {type(e).__name__}: {e}. "
                    "Set CC_CUDA_ALLOW_CPU_FALLBACK=1 to allow CPU fallback "
                    "in dev/test environments."
                )
            self._gpu_ctx = decord.cpu(0)

    def sample(
        self,
        video_path: Optional[Union[Path, str]] = None,
        sample_fps: Optional[float] = None,
    ) -> Iterator[FrameSample]:
        """Yield FrameSample for each selected source frame."""
        for batch in self.sample_batches(batch_size=32, video_path=video_path):
            yield from batch

    def _prepare_loader(self, batch_size: int, video_path) -> tuple:
        """Probe dims and build the decord VideoLoader. Returns (vl, fps, h, w) or (None, fps, h, w)."""
        resolved = Path(video_path) if video_path else self.video_path
        if resolved is None:
            raise ValueError("video_path must be provided")

        # decord's bridge is threading.local(): the module-level set_bridge('torch')
        # ran on the import (main) thread only. The Phase-B prefetch in
        # _run_cuda_inference iterates this sampler from a daemon producer thread,
        # which gets the default 'native' bridge and makes VideoReader/VideoLoader
        # return decord NDArray (no .cpu()). Re-assert the bridge in whatever thread
        # actually decodes — idempotent, cheap, and safe on the main thread too.
        decord.bridge.set_bridge('torch')

        # Probe video dimensions with a lightweight CPU reader
        probe = decord.VideoReader(str(resolved), ctx=decord.cpu(0))
        total = len(probe)
        fps = probe.get_avg_fps() or 30.0
        # With torch bridge: probe[0] returns torch.Tensor; convert to numpy
        # via .cpu().numpy() (no-op move since ctx=cpu).
        first = probe[0].cpu().numpy()
        h, w = first.shape[:2]
        self.last_source_fps = fps
        self.last_selected_frame_count = max(1, (total + self.stride - 1) // self.stride)
        del probe

        if total == 0:
            return None, fps, h, w

        # interval=stride-1: 0 → every frame, 1 → every 2nd frame, etc.
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
        # decord VideoLoader returns batch_indices as (N, 2) — pairs of
        # [video_idx, frame_idx]. We only have one video (vl was given a
        # single path), so take the frame_idx column. The previous
        # reshape(-1) doubled the length to 2N and caused an IndexError
        # when we tried to align with frames_np (length N).
        idx_arr = batch_indices.cpu().numpy()
        if idx_arr.ndim == 2:
            return idx_arr[:, 1].astype(int)
        return idx_arr.reshape(-1).astype(int)

    def sample_batches(
        self,
        batch_size: int = 32,
        video_path: Optional[Union[Path, str]] = None,
    ) -> Iterator[list]:
        """Yield lists of FrameSample using VideoLoader for continuous GPU streaming.

        VideoLoader handles batch management internally. Only batch_size frames
        are in RAM at once regardless of video length.
        """
        vl, fps, h, w = self._prepare_loader(batch_size, video_path)
        if vl is None:
            return
        for batch_data, batch_indices in vl:
            # Torch bridge: VideoLoader yields torch.Tensor on the loader's ctx
            # (cuda when NVDEC-enabled). Convert to numpy at this boundary so
            # downstream consumers (triage, OpenCV-based scoring) keep their
            # numpy contract. A future optimization is to push torch tensors
            # all the way through to kornia and skip this copy entirely.
            frames_np = batch_data.cpu().numpy()                 # (N, H, W, 3)
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

    def sample_gpu_batches(
        self,
        batch_size: int = 32,
        thumbnail_width: int = 640,
        video_path: Optional[Union[Path, str]] = None,
    ) -> Iterator[tuple]:
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
