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
except ImportError:
    decord = None  # type: ignore[assignment]

from card_capture.models import FrameSample


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
        try:
            self._gpu_ctx = _probe_gpu()
        except Exception:
            if not allow_fallback:
                raise RuntimeError(
                    "CudaSampler requires NVDEC (decord GPU context). "
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

    def sample_batches(
        self,
        batch_size: int = 32,
        video_path: Optional[Union[Path, str]] = None,
    ) -> Iterator[list]:
        """Yield lists of FrameSample using VideoLoader for continuous GPU streaming.

        VideoLoader handles batch management internally. Only batch_size frames
        are in RAM at once regardless of video length.
        """
        resolved = Path(video_path) if video_path else self.video_path
        if resolved is None:
            raise ValueError("video_path must be provided")

        # Probe video dimensions with a lightweight CPU reader
        probe = decord.VideoReader(str(resolved), ctx=decord.cpu(0))
        total = len(probe)
        fps = probe.get_avg_fps() or 30.0
        first = probe[0].asnumpy()
        h, w = first.shape[:2]
        self.last_source_fps = fps
        self.last_selected_frame_count = max(1, (total + self.stride - 1) // self.stride)
        del probe

        if total == 0:
            return

        # interval=stride-1: 0 → every frame, 1 → every 2nd frame, etc.
        vl = decord.VideoLoader(
            [str(resolved)],
            ctx=[self._gpu_ctx],
            shape=(batch_size, h, w, 3),
            interval=max(0, self.stride - 1),
            skip=0,
            shuffle=0,
        )

        for batch_data, batch_indices in vl:
            frames_np = batch_data.asnumpy()                     # (N, H, W, 3)
            indices_flat = batch_indices.asnumpy().reshape(-1).astype(int)

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
