"""NVDEC-accelerated stride sampler for the CUDA pipeline.

Replaces the AdaptivePresenceSampler on vast.ai GPU instances.
No presence classifier, no windowing — uniform stride across the full video
with dense coverage for the opening window (catching cards on stands).

GPU-or-die: raises RuntimeError if NVDEC is unavailable and
CC_CUDA_ALLOW_CPU_FALLBACK is not set. The vastai_worker never sets that
variable; it is only for local dev/test environments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np

from card_capture.models import FrameSample


def _probe_gpu() -> object:
    """Return a decord GPU context (index 0), or raise on failure."""
    import decord
    return decord.gpu(0)


class CudaSampler:
    """Uniform-stride video sampler using decord NVDEC for GPU decode.

    Args:
        video_path: Source video file.
        stride: Decode every Nth source frame outside the opening window.
            Default 2 = 30fps effective from a 60fps source.
        opening_scan_s: Always include every frame from the first N seconds,
            regardless of stride, so cards resting on a stand at the start
            are never missed.
    """

    def __init__(
        self,
        video_path: Optional[Union[Path, str]] = None,
        stride: int = 2,
        opening_scan_s: float = 2.0,
    ) -> None:
        self.video_path = Path(video_path) if video_path else None
        self.stride = max(1, stride)
        self.opening_scan_s = max(0.0, opening_scan_s)
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
            import decord
            self._gpu_ctx = decord.cpu(0)

    def sample(
        self,
        video_path: Optional[Union[Path, str]] = None,
        sample_fps: Optional[float] = None,
    ) -> Iterator[FrameSample]:
        """Yield FrameSample for each selected source frame."""
        import decord

        resolved = Path(video_path) if video_path else self.video_path
        if resolved is None:
            raise ValueError("video_path must be provided")

        vr = decord.VideoReader(str(resolved), ctx=self._gpu_ctx)
        total = len(vr)
        self.last_source_fps = vr.get_avg_fps() or 30.0

        # Opening window: every frame (dense — catches cards on stands)
        opening_count = int(self.last_source_fps * self.opening_scan_s)
        opening_indices = list(range(0, min(opening_count, total)))

        # Remaining: every stride-th frame
        stride_indices = list(range(opening_count, total, self.stride))

        # Merge and deduplicate (opening may overlap stride start)
        all_indices = sorted(set(opening_indices + stride_indices))
        self.last_selected_frame_count = len(all_indices)

        if not all_indices:
            return

        # Batch-decode via NVDEC
        frames = vr.get_batch(all_indices)   # shape: (N, H, W, C), GPU or CPU tensor

        for i, idx in enumerate(all_indices):
            frame_np = frames[i].asnumpy()   # transfer to CPU NumPy for downstream compat
            h, w = frame_np.shape[:2]
            ts_ms = int(idx * 1000 / self.last_source_fps)
            yield FrameSample(
                frame_index=idx,
                timestamp_ms=ts_ms,
                image=frame_np,
                width=w,
                height=h,
            )
