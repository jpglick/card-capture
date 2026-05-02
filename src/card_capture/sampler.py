from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .models import FrameSample


@dataclass
class StableWindow:
    """A contiguous run of low-motion frames found during pass 1.

    start_frame, end_frame, and best_frame_index are retained for
    logging/debugging.  frame_candidates holds every (source_frame_index,
    laplacian_variance) pair recorded during the scan so that pass 2 can yield
    multiple candidate frames spread across the window.

    All frame index values are source video frame numbers (0-based) usable
    directly with cv2.CAP_PROP_POS_FRAMES.
    """

    start_frame: int
    end_frame: int
    best_frame_index: int
    frame_candidates: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class DetectionWindow:
    """A contiguous run of detection-rich frames (card confidences > threshold).

    Detection windows are found by scanning with YOLO at low resolution. The
    best frame in each window (highest confidence) is marked for full-res
    processing. All frame indices are source video frame numbers.
    """

    start_frame: int
    end_frame: int
    best_frame_index: int
    best_confidence: float
    frame_detections: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class PresenceWindow:
    """Represents a window where a card is detected (high color variance).
    
    Attributes:
        start_frame: Frame index where presence begins
        end_frame: Frame index where presence ends (inclusive)
        frame_candidates: List of (frame_index, sharpness_score) tuples, sorted by sharpness descending
    """
    start_frame: int
    end_frame: int
    frame_candidates: list[tuple[int, float]] = field(default_factory=list)


class StabilityBasedSampler:
    """Two-pass sampler: cheap diff scan to find still windows, then seek to
    multiple candidate frames within each window for full-resolution detection.

    Pass 1: decode at scan_fps with frames downscaled to scan_width wide.
            Compute per-frame pixel diff; track stable runs. Record the source
            frame number and Laplacian variance for each sampled frame — source
            frame numbers (not scan counters) are stored so they can be passed
            directly to cv2.CAP_PROP_POS_FRAMES.

    Pass 2: for each stable window, yield up to candidates_per_window frames
            evenly distributed across the window's duration. Spreading
            candidates temporally gives the detector a better chance of finding
            a frame where the card is clearly visible, rather than committing to
            the single sharpest moment.

    sample_fps is intentionally ignored — scan_fps is set via the constructor.
    The argument exists solely for interface compatibility with VideoSampler.
    """

    def __init__(
        self,
        scan_fps: float = 10.0,
        scan_width: int = 160,
        motion_threshold: float = 8.0,
        min_stable_frames: int = 5,
        candidates_per_window: int = 5,
    ) -> None:
        self.scan_fps = scan_fps
        self.scan_width = scan_width
        self.motion_threshold = motion_threshold
        self.min_stable_frames = min_stable_frames
        self.candidates_per_window = max(1, candidates_per_window)

    def _flush_run(
        self,
        run_start: Optional[int],
        run_frames: List[Tuple[int, float]],
        windows: List[StableWindow],
    ) -> None:
        """Flush a completed stable run into windows if it meets min_stable_frames."""
        if run_start is not None and len(run_frames) >= self.min_stable_frames:
            best_idx = max(run_frames, key=lambda x: x[1])[0]
            windows.append(
                StableWindow(
                    start_frame=run_start,
                    end_frame=run_frames[-1][0],
                    best_frame_index=best_idx,
                    frame_candidates=list(run_frames),
                )
            )

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
        run_frames: List[Tuple[int, float]] = []
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
                            self._flush_run(run_start, run_frames, windows)
                            run_start = None
                            run_frames = []

                    prev_gray = gray

                frame_index += 1

            # Flush any open run at end of video
            self._flush_run(run_start, run_frames, windows)
        finally:
            capture.release()

        return windows

    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:  # noqa: ARG002
        """Yield up to candidates_per_window FrameSamples per stable window.

        Candidates are selected by evenly spacing across the window's recorded
        frames, giving temporal coverage so the detector can find whichever
        moment the card is most clearly visible.

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
                candidates = window.frame_candidates
                n = len(candidates)
                k = self.candidates_per_window
                if n <= k:
                    selected = candidates
                else:
                    # Evenly distribute k indices across the n candidates
                    step = n / k
                    selected = [candidates[int(i * step)] for i in range(k)]

                for frame_idx, _lap_var in selected:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    # Read timestamp BEFORE capture.read() — OpenCV advances the
                    # position counter on read, which would cause an off-by-one.
                    timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    height, width = frame.shape[:2]
                    yield FrameSample(
                        frame_index=frame_idx,
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


class DetectionGuidedSampler:
    """Two-pass sampler: scan with YOLO at low resolution to find card-present
    windows, then yield candidate frames from those windows for full-res processing.

    Pass 1: decode at scan_fps, downscale to scan_width, run YOLO with conf
            threshold; track windows where detections (confidences > threshold)
            are found. This finds when the card is actually present, not just
            when the camera is steady.

    Pass 2: for each detection window, yield up to candidates_per_window frames
            evenly distributed within the window at full resolution.

    Unlike StabilityBasedSampler (which prioritizes low motion), this prioritizes
    card visibility. Trades more computation (YOLO at scan time) for better
    candidate selection (frames are guaranteed to show cards).
    """

    def __init__(
        self,
        scan_fps: float = 3.0,
        scan_width: int = 160,
        detection_confidence: float = 0.25,
        min_detection_frames: int = 3,
        candidates_per_window: int = 3,
        device: str = "auto",
    ) -> None:
        self.scan_fps = scan_fps
        self.scan_width = scan_width
        self.detection_confidence = detection_confidence
        self.min_detection_frames = min_detection_frames
        self.candidates_per_window = max(1, candidates_per_window)
        self.device = device

    def _find_detection_windows(self, video_path: Path) -> List[DetectionWindow]:
        """Pass 1: scan video with YOLO and return detection window descriptors."""
        from .detectors import CardcaptorUltralyticsDetector

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or self.scan_fps
        frame_step = max(1, int(round(source_fps / self.scan_fps)))
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=self.detection_confidence,
            detection_width=self.scan_width,
            device=self.device,
        )

        windows: List[DetectionWindow] = []
        run_start: Optional[int] = None
        run_detections: List[Tuple[int, float]] = []
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % frame_step == 0:
                    h, w = frame.shape[:2]
                    fs = FrameSample(
                        frame_index=frame_index,
                        timestamp_ms=0,
                        image=frame,
                        width=w,
                        height=h,
                    )
                    detections = detector.detect(fs)
                    if detections:
                        best_conf = max(d.confidence for d in detections)
                        if run_start is None:
                            run_start = frame_index
                        run_detections.append((frame_index, best_conf))
                    else:
                        self._flush_detection_run(run_start, run_detections, windows)
                        run_start = None
                        run_detections = []

                frame_index += 1

            self._flush_detection_run(run_start, run_detections, windows)
        finally:
            capture.release()

        return windows

    def _flush_detection_run(
        self,
        run_start: Optional[int],
        run_detections: List[Tuple[int, float]],
        windows: List[DetectionWindow],
    ) -> None:
        """Flush a completed detection run if it meets min_detection_frames."""
        if run_start is not None and len(run_detections) >= self.min_detection_frames:
            best_idx, best_conf = max(run_detections, key=lambda x: x[1])
            windows.append(
                DetectionWindow(
                    start_frame=run_start,
                    end_frame=run_detections[-1][0],
                    best_frame_index=best_idx,
                    best_confidence=best_conf,
                    frame_detections=list(run_detections),
                )
            )

    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:  # noqa: ARG002
        """Yield candidate frames from detection windows at full resolution.

        sample_fps is intentionally unused — scan_fps is set via the constructor.
        """
        video_path = Path(video_path)
        windows = self._find_detection_windows(video_path)
        if not windows:
            return

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {video_path}")

        try:
            for window in windows:
                detections = window.frame_detections
                n = len(detections)
                k = self.candidates_per_window
                if n <= k:
                    selected = detections
                else:
                    step = n / k
                    selected = [detections[int(i * step)] for i in range(k)]

                for frame_idx, _conf in selected:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    height, width = frame.shape[:2]
                    yield FrameSample(
                        frame_index=frame_idx,
                        timestamp_ms=timestamp_ms,
                        image=frame,
                        width=width,
                        height=height,
                    )
        finally:
            capture.release()


class ContrastBasedSampler:
    """
    Fast, non-ML frame selection using color variance for presence detection.
    
    Pass 1: Scan video at low resolution to detect presence windows (high variance = card present).
    Pass 2: Within each window, rank frames by Laplacian sharpness and yield top candidates.
    """

    def __init__(
        self,
        video_path: str,
        scan_fps: float = 5.0,
        scan_width: int = 160,
        contrast_threshold: float = 1000.0,
        min_presence_frames: int = 3,
        candidates_per_window: int = 3,
    ):
        """
        Args:
            video_path: Path to input video file
            scan_fps: Frames per second to scan in Pass 1 (lower = faster)
            scan_width: Width to downscale for variance computation (lower = faster)
            contrast_threshold: Minimum color variance to consider frame as card-present
            min_presence_frames: Minimum consecutive frames to form a presence window
            candidates_per_window: Number of sharpest frames to yield per window
        """
        self.video_path = video_path
        self.scan_fps = scan_fps
        self.scan_width = scan_width
        self.contrast_threshold = contrast_threshold
        self.min_presence_frames = min_presence_frames
        self.candidates_per_window = candidates_per_window

    def _find_presence_windows(self) -> list[PresenceWindow]:
        """
        Pass 1: Scan video at low resolution to find windows with high color variance.
        
        Returns:
            List of PresenceWindow objects with frame ranges (no candidates yet).
        """
        windows = []
        cap = cv2.VideoCapture(self.video_path)
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = max(1, int(fps / self.scan_fps))
            frame_index = 0
            in_presence_window = False
            window_start = 0
            presence_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample at scan_fps
                if frame_index % frame_skip == 0:
                    # Downscale and compute variance
                    small = cv2.resize(frame, (self.scan_width, int(frame.shape[0] * self.scan_width / frame.shape[1])))
                    variance = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

                    if variance > self.contrast_threshold:
                        if not in_presence_window:
                            in_presence_window = True
                            window_start = frame_index
                            presence_frames = 1
                        else:
                            presence_frames += 1
                    else:
                        if in_presence_window:
                            if presence_frames >= self.min_presence_frames:
                                windows.append(PresenceWindow(
                                    start_frame=window_start,
                                    end_frame=frame_index - frame_skip
                                ))
                            in_presence_window = False

                frame_index += 1

            # Handle window at end of video
            if in_presence_window and presence_frames >= self.min_presence_frames:
                windows.append(PresenceWindow(
                    start_frame=window_start,
                    end_frame=frame_index - 1
                ))

        finally:
            cap.release()

        return windows

    def _score_sharpness_in_window(self, window: PresenceWindow) -> PresenceWindow:
        """
        Pass 2: For frames in a presence window, compute Laplacian sharpness and rank.
        
        Updates window.frame_candidates with (frame_index, sharpness_score) tuples,
        sorted by sharpness descending.
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame)
        
        try:
            frame_scores = []
            for frame_idx in range(window.start_frame, window.end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                frame_scores.append((frame_idx, sharpness))
            
            # Sort by sharpness descending, take top N candidates
            frame_scores.sort(key=lambda x: x[1], reverse=True)
            window.frame_candidates = frame_scores[:self.candidates_per_window]
            
        finally:
            cap.release()
        
        return window

    def sample(self) -> Iterator[PresenceWindow]:
        """
        Generate PresenceWindow objects with ranked frame candidates.
        
        Yields:
            PresenceWindow objects with frame_candidates populated
        """
        presence_windows = self._find_presence_windows()
        
        for window in presence_windows:
            window = self._score_sharpness_in_window(window)
            if window.frame_candidates:
                yield window
