from __future__ import annotations

import math
import time
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from ..models import FrameSample
from ..ingestion import _resolve_reader_backend, _open_capture
from ..presence.classifier import PresenceClassifier as _PresenceClassifier
from ..gpu_utils import (
    compute_variance_gpu,
    compute_sharpness_gpu,
    compute_motion_gpu,
    is_histogram_outlier,
    compute_edge_density_gpu,
    compute_presence_metrics_batched,
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


@dataclass
class _AdaptiveScanFrame:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray
    metrics: dict[str, float]
    motion: float = 0.0
    presence_score: float = 0.0
    delta_score: float = 0.0

    @property
    def sobel_score(self) -> float:
        """Alias for metrics['edge_density'], used by valley-split logic (matches _ScanFrame API)."""
        return self.metrics.get("edge_density", 0.0)


@dataclass
class _ScanFrame:
    """Lightweight scan frame used in fast-scan Pass 1 (no full metrics, no image storage)."""
    frame_index: int
    timestamp_ms: float
    sobel_score: float
    delta_score: float = 0.0


def _valley_split_merge_window_frames(
    frame_indices: list[int],
    valley_min_width_frames: int,
) -> int:
    diffs = [
        curr - prev
        for prev, curr in zip(frame_indices, frame_indices[1:])
        if curr > prev
    ]
    if not diffs:
        return max(1, valley_min_width_frames)
    scan_stride = max(1, int(round(float(np.median(diffs)))))
    merge_window_scan_frames = max(5, valley_min_width_frames * 2)
    return scan_stride * merge_window_scan_frames


def _coalesce_nearby_split_frames(
    split_frames: list[int],
    merge_window_frames: int,
) -> list[int]:
    if not split_frames:
        return []
    coalesced = [split_frames[0]]
    for frame_index in split_frames[1:]:
        if frame_index - coalesced[-1] <= merge_window_frames:
            continue
        coalesced.append(frame_index)
    return coalesced


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
        capture = _open_capture(video_path)
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
        capture = _open_capture(video_path)
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
        capture = _open_capture(video_path)
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
        # Yield 10 identical frames so the tracker can form a stable track
        for i in range(10):
            image = np.zeros((120, 90, 3), dtype=np.uint8)
            image[15:105, 10:80] = 180
            yield FrameSample(frame_index=i, timestamp_ms=i * 100, image=image, width=90, height=120)

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
                from ..gpu_utils import get_device
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
        cap = _open_capture(self.video_path)
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
        cap = _open_capture(self.video_path)
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
        capture = _open_capture(self.video_path)
        try:
            for window in scored_windows:
                for frame_index, _ in window.frame_candidates:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    ok, frame = capture.read()
                    if not ok: continue
                    yield FrameSample(frame_index=frame_index, timestamp_ms=timestamp_ms, image=frame, width=frame.shape[1], height=frame.shape[0])
        finally: capture.release()


class AdaptivePresenceSampler:
    """Two-pass sampler that adapts per video instead of using fixed triage knobs."""

    def __init__(
        self,
        video_path: Optional[Union[Path, str]] = None,
        reader_backend: str = "auto",
        scan_fps: float = 12.0,
        scan_width: int = 192,
        device: str = "auto",
        min_presence_frames: int = 2,
        window_merge_gap: int = 3,
        min_candidates_per_window: int = 3,
        max_candidates_per_window: int = 48,
        empty_pixel_threshold: float = 8.0,
        sobel_threshold: float = 50.0,
        presence_weights_path: Optional[Path] = None,
        presence_threshold: float = 0.5,
        fast_scan_fps: float = 15.0,
        confirm_scan_fps: Optional[float] = None,
        valley_drop_ratio: float = 0.40,
        valley_min_width_frames: int = 3,
        delta_spike_ratio: float = 0.50,
    ):
        self.video_path = str(video_path) if video_path is not None else None
        self.reader_backend = _resolve_reader_backend(reader_backend)
        # scan_fps is the legacy param; fast_scan_fps overrides it for Pass 1
        self.scan_fps = scan_fps
        self.fast_scan_fps = fast_scan_fps
        # confirm_scan_fps defaults to scan_fps for backward compat
        self.confirm_scan_fps = confirm_scan_fps if confirm_scan_fps is not None else scan_fps
        self.valley_drop_ratio = valley_drop_ratio
        self.valley_min_width_frames = valley_min_width_frames
        self.delta_spike_ratio = delta_spike_ratio
        self.last_valley_splits: list[int] = []
        self.scan_width = scan_width
        self.device = device
        self.min_presence_frames = max(1, min_presence_frames)
        self.window_merge_gap = max(0, window_merge_gap)
        self.min_candidates_per_window = max(1, min_candidates_per_window)
        self.max_candidates_per_window = max(
            self.min_candidates_per_window, max_candidates_per_window
        )
        self.empty_pixel_threshold = empty_pixel_threshold
        self.sobel_threshold = sobel_threshold
        self.presence_weights_path = presence_weights_path
        self.presence_threshold = max(0.0, min(1.0, presence_threshold))
        self._presence_classifier: Optional[_PresenceClassifier] = None
        self._scan_frames: list[_AdaptiveScanFrame] = []
        self.last_scan_frame_count = 0
        self.last_presence_window_count = 0
        self.last_selected_frame_count = 0
        self.last_score_threshold: float = 0.0
        self.last_fallback_used = False
        self.last_inter_window_gaps_frames: list[int] = []
        self.last_source_fps: float = 30.0
        self.background_proxies: list[np.ndarray] = []
        self._max_bg_proxies = 5
        self._bg_safety_threshold = 0.4  # If min score > this, we have no empty stand

    @staticmethod
    def _robust_zscores(values: list[float]) -> list[float]:
        if not values:
            return []
        arr = np.asarray(values, dtype=np.float32)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        # Use a more conservative floor to prevent noise-driven explosions
        # 0.02 * (abs(median) + 1.0) provides a dynamic floor relative to the signal level
        scale = max(0.02 * (abs(median) + 1.0), mad * 1.4826)
        z_scores = [float((value - median) / scale) for value in arr]
        # Clip to prevent extreme outliers from drowing out other metrics
        return list(np.clip(z_scores, -10.0, 10.0))

    @staticmethod
    def _otsu_threshold(values: list[float]) -> float:
        if not values:
            return float("inf")
        arr = np.asarray(values, dtype=np.float32)
        if float(arr.max() - arr.min()) <= 1e-3:
            return float(arr.min() - 1e-3) # Everything is active if no variation

        # Clip for histogram binning to avoid issues with extreme tails
        p1, p99 = np.percentile(arr, [1, 99])
        clipped = np.clip(arr, p1, p99)

        bins = max(8, min(64, int(math.sqrt(len(arr))) * 4))
        hist, bin_edges = np.histogram(clipped, bins=bins)
        total = hist.sum()
        if total <= 0:
            return float(np.median(arr))

        prob = hist.astype(np.float64) / float(total)
        cumulative_prob = np.cumsum(prob)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        cumulative_mean = np.cumsum(prob * bin_centers)
        global_mean = cumulative_mean[-1]
        denominator = cumulative_prob * (1.0 - cumulative_prob)
        denominator[denominator == 0.0] = np.nan
        between_class = (global_mean * cumulative_prob - cumulative_mean) ** 2 / denominator

        if np.all(np.isnan(between_class)):
            return float(np.median(arr))

        # Soften threshold by picking the peak index directly rather than the edge after it
        best_index = int(np.nanargmax(between_class))
        return float(bin_edges[best_index])

    def _resize_for_scan(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.scan_width:
            return frame
        scaled_height = max(1, int(round(height * self.scan_width / width)))
        return cv2.resize(frame, (self.scan_width, scaled_height))

    def _score_records(self, records: list[_AdaptiveScanFrame]) -> list[float]:
        sharpness = [record.metrics["sharpness"] for record in records]
        edge_density = [record.metrics["edge_density"] for record in records]
        variance = [record.metrics["variance"] for record in records]
        empty_ratio = [record.metrics["empty_ratio"] for record in records]
        motion = [record.motion for record in records]

        z_sharpness = self._robust_zscores(sharpness)
        z_edge = self._robust_zscores(edge_density)
        z_variance = self._robust_zscores(variance)
        z_empty = self._robust_zscores(empty_ratio)
        z_motion = self._robust_zscores(motion)

        scores: list[float] = []
        for idx in range(len(records)):
            # Robust texture/motion-focused scoring
            score = (
                0.30 * z_sharpness[idx]
                + 0.35 * z_edge[idx]
                + 0.10 * z_variance[idx]
                + 0.25 * z_motion[idx]
                - 0.60 * z_empty[idx]
            )
            scores.append(float(score))
        return scores

    def _scan_video(self, video_path: Path) -> list[_AdaptiveScanFrame]:
        import time
        start_time = time.time()
        sampler = VideoSampler(reader_backend=self.reader_backend)
        device = self.device
        effective_batch_size = 32
        try:
            import torch

            resolved_device = torch.device(device) if device != "auto" else torch.device("cpu")
            if device == "auto":
                try:
                    from ..gpu_utils import get_device

                    resolved_device = get_device()
                except Exception:
                    resolved_device = torch.device("cpu")
            if resolved_device.type == "cpu":
                effective_batch_size = 8
            else:
                effective_batch_size = 32
            device = resolved_device
        except Exception:
            effective_batch_size = 8

        raw_samples = sampler.sample(video_path, self.fast_scan_fps)
        cap = _open_capture(video_path)
        self.last_source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        records: list[_AdaptiveScanFrame] = []
        previous_gray: Optional[np.ndarray] = None
        previous_img: Optional[np.ndarray] = None
        batch_samples: list[FrameSample] = []
        batch_images: list[np.ndarray] = []

        def flush_batch() -> None:
            nonlocal previous_gray, previous_img
            if not batch_images:
                return
            batch_metrics = compute_presence_metrics_batched(
                batch_images,
                device=device,
                batch_size=len(batch_images),
                sobel_threshold=self.sobel_threshold,
                empty_pixel_threshold=self.empty_pixel_threshold,
            )
            for sample, scan_image, metrics in zip(batch_samples, batch_images, batch_metrics):
                gray = cv2.cvtColor(scan_image, cv2.COLOR_BGR2GRAY) if scan_image.ndim == 3 else scan_image
                if previous_gray is None:
                    motion = 0.0
                else:
                    motion = float(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32)).mean())
                previous_gray = gray
                # Compute delta_score as mean absolute pixel diff vs previous frame
                img_f32 = scan_image.astype(np.float32)
                if previous_img is None:
                    delta_score = 0.0
                else:
                    delta_score = float(np.mean(np.abs(img_f32 - previous_img)))
                previous_img = img_f32
                records.append(
                    _AdaptiveScanFrame(
                        frame_index=sample.frame_index,
                        timestamp_ms=sample.timestamp_ms,
                        image=scan_image,
                        metrics=metrics,
                        motion=motion,
                        delta_score=delta_score,
                    )
                )
            batch_samples.clear()
            batch_images.clear()

        for sample in raw_samples:
            batch_samples.append(sample)
            batch_images.append(self._resize_for_scan(sample.image))
            if len(batch_images) >= effective_batch_size:
                flush_batch()
        flush_batch()
        print(f"AdaptivePresenceSampler._scan_video took {time.time() - start_time:.2f} seconds.")
        return records

    def _build_windows(self, records: list[_AdaptiveScanFrame], forced_splits: Optional[list[int]] = None) -> list[PresenceWindow]:
        if not records:
            self.last_inter_window_gaps_frames = []
            return []

        forced = set(forced_splits or [])

        threshold = self.presence_threshold  # classifier path; overwritten by Otsu in fallback
        if self.presence_weights_path is not None:
            # Lazy-load inside subprocess to avoid pickling MPS/CUDA tensors across
            # the multiprocessing boundary.
            if self._presence_classifier is None:
                self._presence_classifier = _PresenceClassifier(
                    weights_path=self.presence_weights_path, device=self.device
                )
            # Use the visual classifier on each scan-resolution proxy frame.
            # Chunk to bound peak memory (sampler can hold thousands of frames).
            scores: list[float] = []
            chunk_size = 32
            for start in range(0, len(records), chunk_size):
                chunk_frames = [r.image for r in records[start:start + chunk_size]]
                scores.extend(self._presence_classifier.score_batch(chunk_frames))
            for record, score in zip(records, scores):
                record.presence_score = score
            active_flags = [s >= self.presence_threshold for s in scores]
        else:
            # Fallback: existing composite z-score path
            scores = self._score_records(records)
            for record, score in zip(records, scores):
                record.presence_score = score
            threshold = self._otsu_threshold(scores)
            edge_vals = np.array([r.metrics["edge_density"] for r in records])
            edge_median = float(np.median(edge_vals))
            edge_mad = float(np.median(np.abs(edge_vals - edge_median)))
            edge_threshold = edge_median + (2.5 * edge_mad * 1.4826) if edge_mad > 1e-6 else float('inf')
            active_flags = []
            for idx, score in enumerate(scores):
                is_otsu_active = score > threshold
                is_feature_active = records[idx].metrics["edge_density"] > edge_threshold
                active_flags.append(is_otsu_active or is_feature_active)
        windows: list[PresenceWindow] = []

        def _flush_window(start: int, end: int) -> None:
            window_records = records[start : end + 1]
            if len(window_records) >= self.min_presence_frames:
                windows.append(
                    PresenceWindow(
                        start_frame=window_records[0].frame_index,
                        end_frame=window_records[-1].frame_index,
                        detection_methods=["adaptive_presence"],
                    )
                )

        start_idx: Optional[int] = None
        last_active_idx: Optional[int] = None
        for idx, is_active in enumerate(active_flags):
            # Forced split: close any open window before processing this frame
            if forced and records[idx].frame_index in forced:
                if start_idx is not None and last_active_idx is not None:
                    _flush_window(start_idx, last_active_idx)
                start_idx = None
                last_active_idx = None

            if is_active:
                if start_idx is None:
                    start_idx = idx
                last_active_idx = idx
                continue
            if start_idx is not None and last_active_idx is not None:
                gap_frames = records[idx].frame_index - records[last_active_idx].frame_index
                if gap_frames <= self.window_merge_gap:
                    continue
                _flush_window(start_idx, last_active_idx)
                start_idx = None
                last_active_idx = None

        if start_idx is not None and last_active_idx is not None:
            _flush_window(start_idx, last_active_idx)

        if windows:
            self.last_presence_window_count = len(windows)
            self.last_score_threshold = threshold
            self.last_fallback_used = False
            self.last_inter_window_gaps_frames = [
                (windows[i + 1].start_frame - windows[i].end_frame)
                for i in range(len(windows) - 1)
            ]
            return windows

        best_idx = int(np.argmax(scores))
        half_span = max(0, self.min_presence_frames // 2)
        start = max(0, best_idx - half_span)
        end = min(len(records) - 1, start + max(self.min_presence_frames, 1) - 1)
        if end - start + 1 < self.min_presence_frames:
            start = max(0, end - self.min_presence_frames + 1)
        window_records = records[start : end + 1]
        if window_records:
            self.last_presence_window_count = 1
            self.last_score_threshold = threshold
            self.last_fallback_used = True
            self.last_inter_window_gaps_frames = []
            return [
                PresenceWindow(
                    start_frame=window_records[0].frame_index,
                    end_frame=window_records[-1].frame_index,
                    detection_methods=["adaptive_fallback"],
                )
            ]
        self.last_inter_window_gaps_frames = []
        return []

    def _candidate_records_for_window(self, window: PresenceWindow) -> list[_AdaptiveScanFrame]:
        return [
            record
            for record in self._scan_frames
            if window.start_frame <= record.frame_index <= window.end_frame
        ]

    def _score_sharpness_in_window(self, window: PresenceWindow) -> PresenceWindow:
        records = self._candidate_records_for_window(window)
        if not records:
            return window

        window_len = len(records)
        # We want to keep as many frames as possible to ensure tracking continuity.
        # Only cap at max_candidates_per_window to prevent memory explosions on very long holds.
        target = min(self.max_candidates_per_window, window_len)

        if window_len <= target:
            window.frame_candidates = [
                (record.frame_index, record.presence_score)
                for record in sorted(records, key=lambda record: record.frame_index)
            ]
            return window

        # If window exceeds max_candidates_per_window, pick frames with highest presence scores
        scored_records = sorted(records, key=lambda record: record.presence_score, reverse=True)
        top_indices = {r.frame_index for r in scored_records[:target]}

        # Ensure we always include start and end of presence for boundary detection
        top_indices.add(records[0].frame_index)
        top_indices.add(records[-1].frame_index)

        selected = sorted(
            (record for record in records if record.frame_index in top_indices),
            key=lambda record: record.frame_index,
        )
        window.frame_candidates = [
            (record.frame_index, record.presence_score) for record in selected
        ]
        return window

    def _decode_selected_frames(self, video_path: Path, frame_indices: list[int]) -> list[FrameSample]:
        if not frame_indices:
            return []

        capture = _open_capture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {video_path}")

        samples: list[FrameSample] = []
        target_indices = set(frame_indices)
        max_target = max(target_indices)

        frame_idx = 0
        try:
            while frame_idx <= max_target:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_idx in target_indices:
                    height, width = frame.shape[:2]
                    timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    samples.append(
                        FrameSample(
                            frame_index=frame_idx,
                            timestamp_ms=timestamp_ms,
                            image=frame,
                            width=width,
                            height=height,
                        )
                    )
                frame_idx += 1
        finally:
            capture.release()
        return samples

    def _resolve_video_path(self, video_path: Optional[Path]) -> Path:
        resolved = video_path or self.video_path
        if resolved is None:
            raise ValueError("video_path must be provided")
        return Path(resolved)

    def _compute_valley_splits(self, scan_frames: list[_AdaptiveScanFrame]) -> list[int]:
        from .valley_splits import find_valley_splits
        from .valley_detection_per_region import find_valley_splits_per_region

        # Global Sobel valley detection
        sobel_scores = [r.sobel_score for r in scan_frames]
        delta_scores = [r.delta_score for r in scan_frames]
        frame_indices = [r.frame_index for r in scan_frames]
        global_splits = find_valley_splits(
            sobel_scores,
            delta_scores,
            frame_indices,
            valley_drop_ratio=self.valley_drop_ratio,
            valley_min_width_frames=self.valley_min_width_frames,
            delta_spike_ratio=self.delta_spike_ratio,
        )

        # Per-region valley detection for multi-card swaps on opposite sides
        proxy_frames = [r.image for r in scan_frames]
        regional_splits = find_valley_splits_per_region(
            proxy_frames,
            grid_rows=3,
            grid_cols=3,
            valley_drop_ratio=0.70,
            min_valley_width_frames=self.valley_min_width_frames,
            frame_indices=frame_indices,
            min_region_votes=5,
        )

        # Combine both signals: global + regional splits
        all_splits = sorted(set(global_splits) | set(regional_splits))
        return _coalesce_nearby_split_frames(
            all_splits,
            merge_window_frames=_valley_split_merge_window_frames(
                frame_indices, self.valley_min_width_frames
            ),
        )

    def _find_presence_windows(self) -> list[PresenceWindow]:
        video_path = self._resolve_video_path(None)
        self._scan_frames = self._scan_video(video_path)
        self.last_valley_splits = self._compute_valley_splits(self._scan_frames)
        return self._build_windows(self._scan_frames, forced_splits=self.last_valley_splits)

    def sample(self, video_path: Path = None, sample_fps: float = None) -> Iterator[FrameSample]:
        resolved_video_path = self._resolve_video_path(video_path)
        self._scan_frames = self._scan_video(resolved_video_path)
        self.last_valley_splits = self._compute_valley_splits(self._scan_frames)
        windows = self._build_windows(self._scan_frames, forced_splits=self.last_valley_splits)

        # Collect background proxies: prioritize frames from detected valley splits
        bg_candidates: list[tuple[float, int, np.ndarray]] = []  # (-score, frame_index, image)
        
        # 1. First, try to pick from valley splits (most reliable empty workspace)
        if self.last_valley_splits:
            # Sort splits to find intervals between them or near them
            # For simplicity, let's just pick frames immediately following split points
            # (which usually represent the start of an empty window)
            for split_idx in self.last_valley_splits[:self._max_bg_proxies]:
                # Find the record for this split or near it
                for record in self._scan_frames:
                    if record.frame_index >= split_idx:
                        heapq.heappush(bg_candidates, (-record.presence_score, record.frame_index, record.image.copy()))
                        break
        
        # 2. If we don't have enough from valleys, pick from lowest presence scores globally
        if len(bg_candidates) < self._max_bg_proxies:
            for record in self._scan_frames:
                score = record.presence_score
                # Be more conservative if we already have some or if score is high
                if score < self._bg_safety_threshold:
                    if len(bg_candidates) < self._max_bg_proxies:
                        heapq.heappush(bg_candidates, (-score, record.frame_index, record.image.copy()))
                    elif score < -bg_candidates[0][0]:
                        heapq.heapreplace(bg_candidates, (-score, record.frame_index, record.image.copy()))

        # Sort by score ascending (lowest first)
        bg_candidates.sort(key=lambda x: -x[0])
        self.background_proxies = [c[2] for c in bg_candidates]

        scored_windows = [self._score_sharpness_in_window(window) for window in windows]
        selected_frame_indices: list[int] = []
        for window in scored_windows:
            selected_frame_indices.extend(frame_index for frame_index, _ in window.frame_candidates)
        deduped_frame_indices = sorted(set(selected_frame_indices))
        self.last_scan_frame_count = len(self._scan_frames)
        self.last_selected_frame_count = len(deduped_frame_indices)
        yield from self._decode_selected_frames(resolved_video_path, deduped_frame_indices)
