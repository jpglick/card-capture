from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from card_capture.core.models import FramePacket


class FrameReader(Protocol):
    def iter_frames(self, video_path: Path | str) -> Iterator[FramePacket]:
        ...


import warnings


def probe_video(path: str | Path) -> dict:
    """Return video metadata (width, height, fps, duration_ms, frame_count)."""
    cap = _open_capture(path)
    try:
        if not cap.isOpened():
            return {}
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((count / fps) * 1000) if fps > 0 else 0
        return {
            "width": w,
            "height": h,
            "fps": fps,
            "frame_count": count,
            "duration_ms": duration_ms
        }
    finally:
        cap.release()


def _open_capture(path: "str | Path") -> cv2.VideoCapture:
    """Open a VideoCapture with hardware acceleration.

    Priority:
    1. VideoToolbox via AVFoundation (macOS native)
    2. HW Acceleration via FFMPEG (Any)
    3. Software fallback
    """
    import platform
    is_macos = platform.system() == "Darwin"

    # 1. macOS Native (AVFoundation)
    if is_macos:
        cap = cv2.VideoCapture(str(path), cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()

    # 2. FFMPEG HW
    cap = cv2.VideoCapture(
        str(path),
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY],
    )
    if cap.isOpened():
        return cap
    cap.release()

    # 3. Software Fallback
    warnings.warn(
        f"Hardware-accelerated video decode unavailable for "
        f"{path!r}; falling back to software decode. Performance will be reduced.",
        RuntimeWarning,
        stacklevel=2,
    )
    return cv2.VideoCapture(str(path))


def _decord_available() -> bool:
    try:
        import decord  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_reader_backend(preferred: str) -> str:
    backend = preferred.strip().lower()
    if backend == "auto":
        return "decord" if _decord_available() else "pyav"
    if backend in {"decord", "pyav", "torchvision", "opencv"}:
        return backend
    raise ValueError("preferred backend must be one of: auto, decord, pyav, torchvision, opencv")


@dataclass
class FrameTriageFilter:
    variance_threshold: float = 25.0
    empty_ratio_threshold: float = 0.98
    blur_threshold: float = 5.0
    empty_pixel_threshold: int = 8

    def evaluate(self, frame: np.ndarray) -> tuple[bool, dict[str, float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        variance = float(gray.var())
        empty_ratio = float((gray <= self.empty_pixel_threshold).mean())

        metrics = {
            "blur": blur,
            "variance": variance,
            "empty_ratio": empty_ratio,
        }
        # v3 wide-funnel behavior: avoid fixed blur/variance gatekeeping and only
        # drop near-empty frames. Relative ranking is handled by RollingWindowTriage.
        accepted = empty_ratio <= self.empty_ratio_threshold
        return accepted, metrics

class RollingWindowTriage:
    def __init__(self, window_size: int = 30, keep_percentile: float = 0.5):
        self.window_size = window_size
        self.keep_percentile = keep_percentile
        self.buffer: list[tuple[int, float]] = []

    def evaluate_score(self, index: int, score: float) -> bool:
        self.buffer.append((index, score))
        if len(self.buffer) < self.window_size:
            return True
        current_buffer = sorted(self.buffer, key=lambda x: x[1], reverse=True)
        keep_count = max(1, int(len(current_buffer) * self.keep_percentile))
        top_indices = {x[0] for x in current_buffer[:keep_count]}
        result = index in top_indices
        self.buffer = [x for x in self.buffer if x[0] > index - self.window_size]
        return result
