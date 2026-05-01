from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .models import FrameSample


class VideoSampler:
    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or sample_fps
        frame_step = max(1, int(round(source_fps / sample_fps))) if sample_fps > 0 else 1
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_step == 0:
                    height, width = frame.shape[:2]
                    timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    yield FrameSample(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        image=frame,
                        width=width,
                        height=height,
                    )
                frame_index += 1
        finally:
            capture.release()


class SyntheticSampler:
    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:
        image = np.zeros((120, 90, 3), dtype=np.uint8)
        image[15:105, 10:80] = 180
        yield FrameSample(
            frame_index=0,
            timestamp_ms=0,
            image=image,
            width=90,
            height=120,
        )
