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

class StabilityBasedSampler:
    def __init__(self, scan_fps: float = 10.0, scan_width: int = 160, motion_threshold: float = 8.0, min_stable_frames: int = 5, candidates_per_window: int = 5):
        self.scan_fps = scan_fps
        self.scan_width = scan_width
        self.motion_threshold = motion_threshold
        self.min_stable_frames = min_stable_frames
        self.candidates_per_window = max(1, candidates_per_window)

    def _flush_run(self, run_start: Optional[int], run_frames: List[Tuple[int, float]], windows: List[StableWindow]):
        if run_start is not None and len(run_frames) >= self.min_stable_frames:
            best_idx = max(run_frames, key=lambda x: x[1])[0]
            windows.append(StableWindow(start_frame=run_start, end_frame=run_frames[-1][0], best_frame_index=best_idx, frame_candidates=list(run_frames)))

    def _find_stable_windows(self, video_path: Path) -> List[StableWindow]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened(): raise ValueError(f"Could not decode video: {video_path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS) or self.scan_fps
        frame_step = max(1, int(round(source_fps / self.scan_fps)))
        windows: List[StableWindow] = []
        run_start: Optional[int] = None
        run_frames: List[Tuple[int, float]] = []
        prev_gray: Optional[np.ndarray] = None
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok: break
                if frame_index % frame_step == 0:
                    h, w = frame.shape[:2]
                    scaled_h = max(1, int(round(h * self.scan_width / w)))
                    small = cv2.resize(frame, (self.scan_width, scaled_h))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if prev_gray is not None:
                        diff = float(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)).mean())
                        if diff < self.motion_threshold:
                            if run_start is None: run_start = frame_index
                            run_frames.append((frame_index, lap_var))
                        else:
                            self._flush_run(run_start, run_frames, windows)
                            run_start = None
                            run_frames = []
                    prev_gray = gray
                frame_index += 1
            self._flush_run(run_start, run_frames, windows)
        finally:
            capture.release()
        return windows

    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:
        video_path = Path(video_path)
        windows = self._find_stable_windows(video_path)
        if not windows: return
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened(): raise ValueError(f"Could not decode video: {video_path}")
        try:
            for window in windows:
                candidates = window.frame_candidates
                n = len(candidates); k = self.candidates_per_window
                selected = candidates if n <= k else [candidates[int(i * (n/k))] for i in range(k)]
                for frame_idx, _lap_var in selected:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    ok, frame = capture.read()
                    if not ok: continue
                    height, width = frame.shape[:2]
                    yield FrameSample(frame_index=frame_idx, timestamp_ms=timestamp_ms, image=frame, width=width, height=height)
        finally:
            capture.release()

class SyntheticSampler:
    def sample(self, video_path: Path, sample_fps: float) -> Iterator[FrameSample]:
        image = np.zeros((120, 90, 3), dtype=np.uint8)
        image[15:105, 10:80] = 180
        yield FrameSample(frame_index=0, timestamp_ms=0, image=image, width=90, height=120)

class ContrastBasedSampler:
    def __init__(self, video_path: str, scan_fps: float = 5.0, scan_width: int = 160, contrast_threshold: float = 600.0, min_presence_frames: int = 3, candidates_per_window: int = 3, device: str = "auto", window_merge_gap: int = 5, motion_threshold: float = 8.0, histogram_sigma: float = 1.5, edge_density_threshold: float = 0.15, sobel_magnitude_threshold: float = 50.0, detection_metrics: list[str] = None):
        self.video_path = video_path
        self.scan_fps = scan_fps
        self.scan_width = scan_width
        self.contrast_threshold = contrast_threshold
        self.min_presence_frames = min_presence_frames
        self.candidates_per_window = candidates_per_window
        self.window_merge_gap = window_merge_gap
        self.motion_threshold = motion_threshold
        self.histogram_sigma = histogram_sigma
        self.edge_density_threshold = edge_density_threshold
        self.sobel_magnitude_threshold = sobel_magnitude_threshold
        self.detection_metrics = detection_metrics if detection_metrics is not None else ["variance"]
        if device == "auto":
            try:
                from .gpu_utils import get_device
                self.device = get_device()
            except ImportError: self.device = torch.device("cpu")
        else: self.device = torch.device(device)

    def _detect_metrics(self, frame_idx: int, frame: np.ndarray, variance: float, motion: float, histogram_stats: tuple[float, float], edge_metrics: tuple[float, bool], enabled_metrics: list[str]) -> list[str]:
        triggered = []
        for metric in enabled_metrics:
            if metric == "variance" and variance > self.contrast_threshold: triggered.append("variance")
            elif metric == "motion" and motion > self.motion_threshold: triggered.append("motion")
            elif metric == "histogram":
                mean, std_dev = histogram_stats
                if is_histogram_outlier(variance, mean, std_dev, sigma_threshold=self.histogram_sigma): triggered.append("histogram")
            elif metric == "edge":
                _, is_high = edge_metrics
                if is_high: triggered.append("edge")
        return triggered

    def _find_presence_windows(self) -> list[PresenceWindow]:
        windows = []
        cap = cv2.VideoCapture(self.video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_skip = max(1, int(fps / self.scan_fps))
            frame_index = 0
            in_presence_window = False; window_start = 0; presence_frames = 0
            window_triggered_methods = set(); prev_frame = None
            while True:
                ret, frame = cap.read()
                if not ret: break
                if frame_index % frame_skip == 0:
                    small = cv2.resize(frame, (self.scan_width, int(frame.shape[0] * self.scan_width / frame.shape[1])))
                    variance = compute_variance_gpu(small, self.device)
                    motion = compute_motion_gpu(prev_frame, small, self.device) if "motion" in self.detection_metrics and prev_frame is not None else 0.0
                    edge_metrics = compute_edge_density_gpu(small, self.device, sobel_threshold=self.sobel_magnitude_threshold, edge_density_threshold=self.edge_density_threshold) if "edge" in self.detection_metrics else (0.0, False)
                    triggered = self._detect_metrics(frame_index, small, variance, motion=motion, histogram_stats=(0.0, 0.0), edge_metrics=edge_metrics, enabled_metrics=self.detection_metrics)
                    if triggered:
                        if not in_presence_window: in_presence_window = True; window_start = frame_index; presence_frames = 1; window_triggered_methods = set(triggered)
                        else: presence_frames += 1; window_triggered_methods.update(triggered)
                    else:
                        if in_presence_window:
                            if presence_frames >= self.min_presence_frames: windows.append(PresenceWindow(start_frame=window_start, end_frame=frame_index - frame_skip, detection_methods=list(window_triggered_methods)))
                            in_presence_window = False; window_triggered_methods = set()
                    prev_frame = small
                frame_index += 1
            if in_presence_window and presence_frames >= self.min_presence_frames: windows.append(PresenceWindow(start_frame=window_start, end_frame=frame_index - 1, detection_methods=list(window_triggered_methods)))
        finally: cap.release()
        return self._merge_nearby_windows(windows, max_gap=self.window_merge_gap)

    def _merge_nearby_windows(self, windows: list[PresenceWindow], max_gap: int = 5) -> list[PresenceWindow]:
        if not windows: return []
        merged = []; current_window = windows[0]
        for next_window in windows[1:]:
            if next_window.start_frame - current_window.end_frame <= max_gap:
                current_window = PresenceWindow(start_frame=current_window.start_frame, end_frame=next_window.end_frame, detection_methods=list(set(current_window.detection_methods + next_window.detection_methods)))
            else: merged.append(current_window); current_window = next_window
        merged.append(current_window)
        return merged

    def _score_sharpness_in_window(self, window: PresenceWindow) -> PresenceWindow:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame)
        try:
            frame_scores = []
            for frame_idx in range(window.start_frame, window.end_frame + 1):
                ret, frame = cap.read()
                if not ret: break
                frame_scores.append((frame_idx, compute_sharpness_gpu(frame, self.device)))
            frame_scores.sort(key=lambda x: x[1], reverse=True)
            window.frame_candidates = frame_scores[:self.candidates_per_window]
        finally: cap.release()
        return window

    def sample(self, video_path: Path = None, sample_fps: float = None) -> Iterator[FrameSample]:
        presence_windows = self._find_presence_windows()
        scored_windows = [self._score_sharpness_in_window(w) for w in presence_windows]
        capture = cv2.VideoCapture(self.video_path)
        try:
            for window in scored_windows:
                for frame_index, _ in window.frame_candidates:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    ok, frame = capture.read()
                    if not ok: continue
                    yield FrameSample(frame_index=frame_index, timestamp_ms=timestamp_ms, image=frame, width=frame.shape[1], height=frame.shape[0])
        finally: capture.release()
