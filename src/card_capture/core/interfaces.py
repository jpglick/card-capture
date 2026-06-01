from __future__ import annotations

from typing import Iterable, List, Optional, Protocol, Tuple, Union
from pathlib import Path

import numpy as np

from .models import (
    CardDetection,
    DetectionPacket,
    FramePacket,
    FrameSample,
)


class CardDetector(Protocol):
    """Protocol for card detection backends."""

    def detect(self, frame: FrameSample) -> List[CardDetection]:
        """Detect cards in a single frame."""
        ...

    def detect_batch(
        self,
        frames: list[FramePacket],
        confidence_threshold: float,
        tensor_input: Optional["torch.Tensor"] = None,
    ) -> list[DetectionPacket]:
        """Detect cards in a batch of frames."""
        ...


class FrameSampler(Protocol):
    """Protocol for video sampling strategies."""

    def sample(
        self, video_path: str | Path, sample_fps: float = 0.0, pixel_format: str = "bgr24"
    ) -> Iterable[FrameSample]:
        """Sample frames from a video file."""
        ...
