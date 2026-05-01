from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .models import FrameSample


@dataclass
class StableWindow:
    """A contiguous run of low-motion frames found during pass 1.

    start_frame and end_frame are retained for logging/debugging.
    Only best_frame_index is used by pass 2 when seeking the video capture.
    All three values are source video frame numbers (0-based, usable with
    cv2.CAP_PROP_POS_FRAMES).
    """

    start_frame: int
    end_frame: int
    best_frame_index: int


class StabilityBasedSampler:
    """Two-pass sampler: cheap diff scan to find still windows, then seek to
    the sharpest frame in each window for full-resolution detection.

    Pass 1: decode at scan_fps with frames downscaled to scan_width wide.
            Compute per-frame pixel diff; track stable runs. Record the source
            frame number and Laplacian variance for each sampled frame — source
            frame numbers (not scan counters) are stored so they can be passed
            directly to cv2.CAP_PROP_POS_FRAMES.

    Pass 2: for each stable window, seek to best_frame_index and yield the
            full-resolution FrameSample.

    sample_fps is intentionally ignored — scan_fps is set via the constructor.
    The argument exists solely for interface compatibility with VideoSampler.
    """

    def __init__(
        self,
        scan_fps: float = 10.0,
        scan_width: int = 160,
        motion_threshold: float = 8.0,
        min_stable_frames: int = 5,
    ) -> None:
        self.scan_fps = scan_fps
        self.scan_width = scan_width
        self.motion_threshold = motion_threshold
        self.min_stable_frames = min_stable_frames

    def _find_stable_windows(self, video_path: Path) -> List[StableWindow]:
        """Pass 1: decode at low resolution and return stable window descriptors."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or self.scan_fps
        frame_step = max(1, int(round(source_fps / self.scan_fps)))

        windows: List[StableWindow] = []
        run_start: Optional[int] = None
        # Each entry: (source_frame_index, laplacian_variance)
        run_frames: List[tuple] = []
        prev_gray: Optional[np.ndarray] = None
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % frame_step == 0:
                    h, w = frame.shape[:2]
                    scaled_h = max(1, int(round(h * self.scan_width / w)))
                    small = cv2.resize(frame, (self.scan_width, scaled_h))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                    if prev_gray is not None:
                        diff = float(
                            np.abs(
                                gray.astype(np.float32) - prev_gray.astype(np.float32)
                            ).mean()
                        )
                        if diff < self.motion_threshold:
                            if run_start is None:
                                run_start = frame_index
                            run_frames.append((frame_index, lap_var))
                        else:
                            if (
                                run_start is not None
                                and len(run_frames) >= self.min_stable_frames
                            ):
                                best_idx = max(run_frames, key=lambda x: x[1])[0]
                                windows.append(
                                    StableWindow(
                                        start_frame=run_start,
                                        end_frame=run_frames[-1][0],
                                        best_frame_index=best_idx,
                                    )
                                )
                            run_start = None
                            run_frames = []

                    prev_gray = gray

                frame_index += 1

            # Flush any open run at end of video
            if run_start is not None and len(run_frames) >= self.min_stable_frames:
                best_idx = max(run_frames, key=lambda x: x[1])[0]
                windows.append(
                    StableWindow(
                        start_frame=run_start,
                        end_frame=run_frames[-1][0],
                        best_frame_index=best_idx,
                    )
                )
        finally:
            capture.release()

        return windows

    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:  # noqa: ARG002
        """Yield one FrameSample per stable window (the sharpest frame in each).

        sample_fps is intentionally unused — scan_fps is set via the constructor.
        """
        video_path = Path(video_path)
        windows = self._find_stable_windows(video_path)
        if not windows:
            return

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {video_path}")

        try:
            for window in windows:
                capture.set(cv2.CAP_PROP_POS_FRAMES, window.best_frame_index)
                # Read timestamp BEFORE capture.read() — OpenCV advances the
                # position counter on read, which would cause an off-by-one.
                timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                yield FrameSample(
                    frame_index=window.best_frame_index,
                    timestamp_ms=timestamp_ms,
                    image=frame,
                    width=width,
                    height=height,
                )
        finally:
            capture.release()


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
