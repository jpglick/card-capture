from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from .models import FramePacket


class FrameReader(Protocol):
    def iter_frames(self, video_path: Path | str) -> Iterator[FramePacket]:
        ...


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
    if backend in {"decord", "pyav"}:
        return backend
    raise ValueError("preferred backend must be one of: auto, decord, pyav")


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
