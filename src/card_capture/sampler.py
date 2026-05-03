from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .models import FrameSample
from .ingestion import _resolve_reader_backend
from .gpu_utils import (
    compute_variance_gpu,
    compute_sharpness_gpu,
    compute_motion_gpu,
    is_histogram_outlier,
    compute_edge_density_gpu,
)

@dataclass
class StableWindow:
    start_frame: int
    end_frame: int
    best_frame_index: int
    frame_candidates: List[Tuple[int, float]] = field(default_factory=list)

@dataclass
class DetectionWindow:
    start_frame: int
    end_frame: int
    best_frame_index: int
    best_confidence: float
    frame_detections: List[Tuple[int, float]] = field(default_factory=list)

@dataclass
class PresenceWindow:
    start_frame: int
    end_frame: int
    frame_candidates: list[tuple[int, float]] = field(default_factory=list)
    detection_methods: list[str] = field(default_factory=list)

class VideoSampler:
    def __init__(self, reader_backend: str = "auto"):
        self.reader_backend = _resolve_reader_backend(reader_backend)

    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:
        if self.reader_backend == "decord":
            yield from self._sample_with_decord(video_path, sample_fps)
        elif self.reader_backend == "pyav":
            yield from self._sample_with_pyav(video_path, sample_fps)
        else:
            yield from self._sample_with_cv2(video_path, sample_fps)

    def _sample_with_cv2(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {video_path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS) or sample_fps
        frame_step = max(1, int(round(source_fps / sample_fps))) if sample_fps > 0 else 1
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok: break
                if frame_index % frame_step == 0:
                    height, width = frame.shape[:2]
                    timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    yield FrameSample(frame_index=frame_index, timestamp_ms=timestamp_ms, image=frame, width=width, height=height)
                frame_index += 1
        finally:
            capture.release()

    def _sample_with_decord(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:
        yield from self._sample_with_cv2(video_path, sample_fps)

    def _sample_with_pyav(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:
        yield from self._sample_with_cv2(video_path, sample_fps)
