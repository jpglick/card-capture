"""Worker subsystem for the card-capture Metaflow pipeline.

Contains the multiprocessing producer/consumer machinery (Stages 1–3) that is
shared between the Metaflow step (pipeline/steps/detect.py) and the legacy
monolith path.  The VideoProcessor class and everything it exclusively uses
have been deleted — they lived in the now-retired pipeline.py monolith.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full
from typing import Any, List, Optional

import cv2
import numpy as np

from .ingestion import FrameTriageFilter, RollingWindowTriage
from .models import (
    CornerDetection,
    DetectionPacket,
    FramePacket,
    FrameSample,
    PerformanceTelemetry,
)

# ---------------------------------------------------------------------------
# Queue control / timing constants
# ---------------------------------------------------------------------------

_SENTINEL = "__card_capture_queue_sentinel__"
_QUEUE_POLL_INTERVAL_SECONDS = 0.1
_QUEUE_RETRY_MAX_WAIT_SECONDS = 60.0
_DRAIN_IDLE_TIMEOUT_SECONDS = 300.0


# ---------------------------------------------------------------------------
# Data envelopes and stats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FrameEnvelope:
    frame_packet: FramePacket


@dataclass(frozen=True)
class _DetectionEnvelope:
    detection_packet: DetectionPacket
    source_frame_path: str
    triage_metrics: dict[str, float]


@dataclass(frozen=True)
class _ProducerStats:
    frame_count: int
    accepted_frame_count: int
    accepted_frame_presence: list[tuple[int, int, bool]] = field(default_factory=list)
    sampler_telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ConsumerStats:
    yolo_frames: int
    yolo_batches: int
    yolo_elapsed_s: float
    device_resolved: str


# ---------------------------------------------------------------------------
# ProcessingOptions — trimmed to fields actually used by _run_pipeline_workers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessingOptions:
    output_dir: Path
    reader_backend: str = "auto"
    queue_size: int = 256
    inference_batch_size: int = 16
    corner_confidence_threshold: float = 0.5
    blur_threshold: float = 30.0
    variance_threshold: float = 20.0
    empty_pixel_threshold: float = 0.98
    group_gap_ms: int = 300
    spatial_variance_threshold: float = 300.0
    telemetry_scope: str = "canonical"
    background_frames: int = 30
    background_threshold: float = 15.0
    null_patience_frames: int = 20
    min_track_length: int = 6
    use_kornia: bool = True
    kornia_device: str = "auto"
    triage_keep_percentile: float = 0.05
    rotate_180: bool = False
    tracker_backend: str = "bytetrack"
    centroid_jump_ratio: float = 0.30
    centroid_jump_frames: int = 3
    foil_threshold: float = 50.0
    enable_foil_aware_fusion: bool = True
    
    # Appearance sessionization & back-half
    novelty_floor: float = 0.30
    track_confidence_floor: float = 0.0
    stand_novelty_max: float = 0.065
    stand_sharpness_max: float = 0.092
    use_fb_classifier: bool = True
    laplacian_scan_stride: int = 4
    max_corner_gap_frames: int = 15
    corner_refinement: bool = False
    appearance_same_threshold: float = 0.15
    appearance_change_threshold: float = 0.30
    appearance_confirm_frames: int = 3
    bridge_min_occurrences: int = 3
    bridge_position_ratio: float = 0.80
    bridge_neighbor_change_ratio: float = 0.80
    bridge_novelty_margin: float = 0.05
    bridge_max_length_ratio: float = 0.75


# ---------------------------------------------------------------------------
# NullStateDetector — background-subtraction workspace-empty check
# ---------------------------------------------------------------------------

class NullStateDetector:
    def __init__(self, frames: int = 30, threshold: float = 15.0):
        self.frames = frames
        self.threshold = threshold
        self.background_model = None
        self._bg_model_u8 = None
        self.frame_count = 0

    def warmup_batch(self, frames: list[np.ndarray]) -> None:
        """Initialize background model from a batch of frames."""
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            if self.background_model is None:
                self.background_model = gray.astype(np.float32)
                self.frame_count = 1
            else:
                self.background_model = (
                    (self.background_model * self.frame_count + gray) / (self.frame_count + 1)
                )
                self.frame_count += 1

        if self.background_model is not None:
            self._bg_model_u8 = self.background_model.astype(np.uint8)
            self.frame_count = self.frames  # FORCE ACTIVE

    def is_workspace_empty(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.background_model is None:
            self.background_model = np.zeros_like(gray, dtype=np.float32)

        # Ensure dimensions match for comparison
        target_h, target_w = self.background_model.shape[:2]
        if gray.shape[0] != target_h or gray.shape[1] != target_w:
            gray = cv2.resize(gray, (target_w, target_h))

        if self.frame_count < self.frames:
            self.background_model = (
                (self.background_model * self.frame_count + gray) / (self.frame_count + 1)
            )
            self.frame_count += 1
            return False  # Return False during warmup so we don't accidentally trigger resets

        if self._bg_model_u8 is None:
            self._bg_model_u8 = self.background_model.astype(np.uint8)

        diff = cv2.absdiff(gray, self._bg_model_u8)
        return float(np.mean(diff)) < self.threshold


# ---------------------------------------------------------------------------
# PipelineTimer — lightweight stage timing helper
# ---------------------------------------------------------------------------

class PipelineTimer:
    def __init__(self):
        self.start_time = time.monotonic()
        self.timings: dict[str, float] = {}

    def record(self, stage: str):
        self.timings[stage] = time.monotonic() - self.start_time


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _run_pipeline_workers(
    video_path: Path,
    video_id: int,
    frame_dir: Path,
    sampler,
    detector,
    options: ProcessingOptions,
) -> tuple[_ProducerStats, _ConsumerStats, list[_DetectionEnvelope]]:
    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=options.queue_size)
    detection_queue = ctx.Queue(maxsize=options.queue_size)
    error_queue = ctx.Queue(maxsize=8)
    stats_queue = ctx.Queue(maxsize=1)
    consumer_stats_queue = ctx.Queue(maxsize=1)

    producer = ctx.Process(
        target=_producer_main,
        args=(
            str(video_path),
            video_id,
            sampler,
            options.blur_threshold,
            options.variance_threshold,
            options.empty_pixel_threshold,
            options.background_frames,
            options.background_threshold,
            options.triage_keep_percentile,
            options.output_dir,
            frame_queue,
            stats_queue,
            error_queue,
        ),
        name="producer",
    )
    consumer = ctx.Process(
        target=_consumer_main,
        args=(
            detector,
            options.inference_batch_size,
            options.corner_confidence_threshold,
            options.output_dir,
            video_id,
            frame_queue,
            detection_queue,
            error_queue,
            consumer_stats_queue,
        ),
        name="consumer",
    )
    producer.start()
    consumer.start()

    try:
        detections = _drain_detection_queue(
            detection_queue,
            error_queue,
            producer,
            consumer,
            idle_timeout_s=_DRAIN_IDLE_TIMEOUT_SECONDS,
        )
        producer.join(timeout=10)
        consumer.join(timeout=10)
        _raise_child_error_if_present(error_queue, producer, consumer)
        _raise_on_bad_exitcode(producer)
        _raise_on_bad_exitcode(consumer)
        producer_stats = _read_producer_stats(stats_queue)
        consumer_stats = _read_consumer_stats(consumer_stats_queue)
        return producer_stats, consumer_stats, detections
    except Exception:
        _stop_process(producer)
        _stop_process(consumer)
        producer.join(timeout=2)
        consumer.join(timeout=2)
        raise
    finally:
        frame_queue.close()
        detection_queue.close()
        error_queue.close()
        stats_queue.close()
        consumer_stats_queue.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_producer_stats(stats_queue) -> _ProducerStats:
    try:
        stats = stats_queue.get(timeout=2)
    except Empty as exc:
        raise RuntimeError("producer did not publish frame statistics") from exc
    if isinstance(stats, _ProducerStats):
        return stats
    raise RuntimeError(f"unexpected producer stats payload: {type(stats)!r}")


def _read_consumer_stats(stats_queue) -> _ConsumerStats:
    try:
        stats = stats_queue.get(timeout=5)
    except Empty:
        return _ConsumerStats(yolo_frames=0, yolo_batches=0, yolo_elapsed_s=0.0, device_resolved="unknown")
    if isinstance(stats, _ConsumerStats):
        return stats
    return _ConsumerStats(yolo_frames=0, yolo_batches=0, yolo_elapsed_s=0.0, device_resolved="unknown")


def _drain_detection_queue(
    detection_queue,
    error_queue,
    producer,
    consumer,
    idle_timeout_s: float,
) -> list[_DetectionEnvelope]:
    rows: list[_DetectionEnvelope] = []
    last_activity = time.monotonic()
    while True:
        _raise_child_error_if_present(error_queue, producer, consumer)
        try:
            item = detection_queue.get(timeout=_QUEUE_POLL_INTERVAL_SECONDS)
        except Empty:
            if not producer.is_alive() and not consumer.is_alive():
                _raise_child_error_if_present(error_queue, producer, consumer)
                raise RuntimeError("consumer exited before emitting sentinel")
            if time.monotonic() - last_activity > idle_timeout_s:
                raise RuntimeError("timed out waiting for consumer output or sentinel")
            continue
        last_activity = time.monotonic()
        if item == _SENTINEL:
            return rows
        if not isinstance(item, _DetectionEnvelope):
            raise RuntimeError(f"unexpected detection queue payload: {type(item)!r}")
        rows.append(item)


def _raise_child_error_if_present(error_queue, producer, consumer) -> None:
    try:
        error = error_queue.get_nowait()
    except Empty:
        return
    _stop_process(producer)
    _stop_process(consumer)
    if isinstance(error, dict):
        worker = error.get("worker", "child")
        message = error.get("message", "unknown error")
        tb = error.get("traceback", "")
        detail = f"{worker} process failed: {message}"
        if tb:
            detail = f"{detail}\n{tb}"
        raise RuntimeError(detail)
    raise RuntimeError(f"child process failed with payload: {error!r}")


def _raise_on_bad_exitcode(process: mp.Process) -> None:
    if process.is_alive():
        _stop_process(process)
        raise RuntimeError(f"{process.name} did not exit cleanly before timeout")
    if process.exitcode != 0:
        raise RuntimeError(f"{process.name} exited with code {process.exitcode}")


def _stop_process(process: mp.Process) -> None:
    if process.is_alive():
        process.terminate()


def _put_with_retry(
    q,
    item,
    timeout: float = _QUEUE_POLL_INTERVAL_SECONDS,
    max_wait_s: float = _QUEUE_RETRY_MAX_WAIT_SECONDS,
) -> None:
    if max_wait_s <= 0:
        raise ValueError("max_wait_s must be positive")

    start = time.monotonic()
    backoff = 0.01
    while True:
        try:
            q.put(item, timeout=timeout)
            return
        except Full:
            if time.monotonic() - start >= max_wait_s:
                raise RuntimeError(
                    f"timed out after {max_wait_s:.2f}s while enqueuing control/data payload"
                )
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 0.2)


def _put_or_fail(q, item) -> None:
    _put_with_retry(
        q,
        item,
        timeout=_QUEUE_POLL_INTERVAL_SECONDS,
        max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
    )


# ---------------------------------------------------------------------------
# Producer subprocess
# ---------------------------------------------------------------------------

def _producer_main(
    video_path: str,
    video_id: int,
    sampler,
    blur_threshold: float,
    variance_threshold: float,
    empty_pixel_threshold: float,
    background_frames: int,
    background_threshold: float,
    triage_keep_percentile: float,
    output_dir: Path,
    frame_queue,
    stats_queue,
    error_queue,
) -> None:
    frame_count = 0
    accepted_frame_count = 0
    accepted_frame_presence: list[tuple[int, int, bool]] = []
    triage = FrameTriageFilter(
        variance_threshold=variance_threshold,
        empty_ratio_threshold=empty_pixel_threshold,
        blur_threshold=blur_threshold,
    )
    rolling_triage = RollingWindowTriage(keep_percentile=triage_keep_percentile)
    null_detector = NullStateDetector(frames=background_frames, threshold=background_threshold)

    # Prepare proxy directory
    proxy_dir = output_dir / "bg_proxies"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    saved_proxies = []

    try:
        frames_iter = sampler.sample(Path(video_path), 0.0)

        # Manually trigger first iteration to ensure Pass 1 (Scan) runs in child process
        # and populates background_proxies.
        try:
            first_frame = next(frames_iter)

            proxies = getattr(sampler, "background_proxies", [])
            if proxies:
                null_detector.warmup_batch(proxies)
                for idx, proxy_img in enumerate(proxies):
                    path = (proxy_dir / f"video_{video_id}_proxy_{idx}.jpg").resolve()
                    cv2.imwrite(str(path), proxy_img)
                    saved_proxies.append(str(path))

            # Process all frames from the iterator
            import itertools
            for frame in itertools.chain([first_frame], frames_iter):
                timer = PipelineTimer()
                frame_count += 1

                # Workspace empty check (background subtraction)
                if null_detector.is_workspace_empty(frame.image):
                    continue

                accepted, metrics = triage.evaluate(frame.image)
                timer.record("t_ingest")

                if not accepted:
                    continue

                if not rolling_triage.evaluate_score(frame.frame_index, metrics["blur"]):
                    continue

                accepted_frame_presence.append((frame.frame_index, frame.timestamp_ms, False))
                accepted_frame_count += 1
                t_ingest = timer.timings.get("t_ingest", 0.0)
                t_io = 0.0

                _put_or_fail(
                    frame_queue,
                    _FrameEnvelope(
                        frame_packet=FramePacket(
                            frame_index=frame.frame_index,
                            timestamp_ms=frame.timestamp_ms,
                            image=frame.image,
                            width=frame.width,
                            height=frame.height,
                            triage_metrics={**metrics, "workspace_empty": 0.0},
                            telemetry=PerformanceTelemetry(
                                t_ingest=t_ingest,
                                t_io=t_io,
                            ),
                        ),
                    ),
                )
        except StopIteration:
            pass
    except Exception as exc:
        _put_with_retry(
            error_queue,
            _serialize_error("producer", exc),
            timeout=_QUEUE_POLL_INTERVAL_SECONDS,
            max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
        )
    finally:
        sampler_telemetry: dict[str, Any] = {"sampler_type": type(sampler).__name__}
        if saved_proxies:
            sampler_telemetry["background_proxies_paths"] = saved_proxies

        for attr in (
            "last_scan_frame_count",
            "last_presence_window_count",
            "last_selected_frame_count",
            "last_score_threshold",
            "last_fallback_used",
            "last_inter_window_gaps_frames",
            "last_source_fps",
            "last_valley_splits",
            "target_yolo_fps",
        ):
            value = getattr(sampler, attr, None)
            if value is not None:
                sampler_telemetry[attr] = value
        _put_with_retry(
            stats_queue,
            _ProducerStats(
                frame_count=frame_count,
                accepted_frame_count=accepted_frame_count,
                accepted_frame_presence=accepted_frame_presence,
                sampler_telemetry=sampler_telemetry,
            ),
            timeout=_QUEUE_POLL_INTERVAL_SECONDS,
            max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
        )
        _put_with_retry(
            frame_queue,
            _SENTINEL,
            timeout=_QUEUE_POLL_INTERVAL_SECONDS,
            max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
        )


# ---------------------------------------------------------------------------
# Consumer subprocess
# ---------------------------------------------------------------------------

def _consumer_main(
    detector,
    inference_batch_size: int,
    corner_confidence_threshold: float,
    output_dir: Path,
    video_id: int,
    frame_queue,
    detection_queue,
    error_queue,
    consumer_stats_queue,
) -> None:
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    device_resolved = "unknown"
    if hasattr(detector, "_resolve_device"):
        try:
            device_resolved = detector._resolve_device()
        except Exception:
            pass

    yolo_frames = 0
    yolo_batches = 0
    yolo_elapsed_s = 0.0

    batch: list[_FrameEnvelope] = []
    try:
        while True:
            item = frame_queue.get()
            if item == _SENTINEL:
                if batch:
                    t0 = time.monotonic()
                    _consume_batch(detector, batch, corner_confidence_threshold, frame_dir, video_id, detection_queue)
                    yolo_elapsed_s += time.monotonic() - t0
                    yolo_frames += len(batch)
                    yolo_batches += 1
                _put_with_retry(
                    consumer_stats_queue,
                    _ConsumerStats(
                        yolo_frames=yolo_frames,
                        yolo_batches=yolo_batches,
                        yolo_elapsed_s=yolo_elapsed_s,
                        device_resolved=device_resolved,
                    ),
                    timeout=_QUEUE_POLL_INTERVAL_SECONDS,
                    max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
                )
                _put_with_retry(
                    detection_queue,
                    _SENTINEL,
                    timeout=_QUEUE_POLL_INTERVAL_SECONDS,
                    max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
                )
                return
            if not isinstance(item, _FrameEnvelope):
                raise RuntimeError(f"unexpected frame queue payload: {type(item)!r}")
            batch.append(item)
            if len(batch) >= inference_batch_size:
                t0 = time.monotonic()
                _consume_batch(detector, batch, corner_confidence_threshold, frame_dir, video_id, detection_queue)
                yolo_elapsed_s += time.monotonic() - t0
                yolo_frames += len(batch)
                yolo_batches += 1
                batch = []
    except Exception as exc:
        _put_with_retry(
            error_queue,
            _serialize_error("consumer", exc),
            timeout=_QUEUE_POLL_INTERVAL_SECONDS,
            max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
        )
        _put_with_retry(
            detection_queue,
            _SENTINEL,
            timeout=_QUEUE_POLL_INTERVAL_SECONDS,
            max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
        )


def _consume_batch(
    detector,
    batch: list[_FrameEnvelope],
    corner_confidence_threshold: float,
    frame_dir: Path,
    video_id: int,
    detection_queue,
) -> None:
    if not batch:
        return

    if hasattr(detector, "detect_batch"):
        timer = PipelineTimer()
        frames = [row.frame_packet for row in batch]
        detections = detector.detect_batch(frames, confidence_threshold=corner_confidence_threshold)
        timer.record("t_detect")
        t_detect_per_frame = timer.timings["t_detect"] / len(batch)

        source_by_frame = {
            (row.frame_packet.frame_index, row.frame_packet.timestamp_ms): row for row in batch
        }
        for detection_packet in detections:
            if detection_packet.corner_detection.confidence < corner_confidence_threshold:
                continue
            source = source_by_frame.get((detection_packet.frame_index, detection_packet.timestamp_ms))
            if source is None:
                continue

            # Save the proxy image so that Background Novelty and Tracking can read it.
            # During refinement, this will be overwritten with the high-res frame.
            source_frame_path = (frame_dir / f"video_{video_id}_frame_{source.frame_packet.frame_index}_{source.frame_packet.timestamp_ms}.jpg").resolve()
            if not source_frame_path.exists():
                cv2.imwrite(str(source_frame_path), source.frame_packet.image)

            prod_telemetry = source.frame_packet.telemetry or PerformanceTelemetry()
            telemetry = PerformanceTelemetry(
                t_ingest=prod_telemetry.t_ingest,
                t_io=prod_telemetry.t_io,
                t_detect=t_detect_per_frame,
            )

            _put_or_fail(
                detection_queue,
                _DetectionEnvelope(
                    detection_packet=DetectionPacket(
                        frame_index=detection_packet.frame_index,
                        timestamp_ms=detection_packet.timestamp_ms,
                        width=detection_packet.width,
                        height=detection_packet.height,
                        corner_detection=detection_packet.corner_detection,
                        telemetry=telemetry,
                    ),
                    source_frame_path=str(source_frame_path),
                    triage_metrics=source.frame_packet.triage_metrics,
                ),
            )
        return

    for row in batch:
        legacy_frame = FrameSample(
            frame_index=row.frame_packet.frame_index,
            timestamp_ms=row.frame_packet.timestamp_ms,
            image=row.frame_packet.image,
            width=row.frame_packet.width,
            height=row.frame_packet.height,
        )
        timer = PipelineTimer()
        detections = list(detector.detect(legacy_frame))
        timer.record("t_detect")
        t_detect = timer.timings["t_detect"]

        prod_telemetry = row.frame_packet.telemetry or PerformanceTelemetry()
        telemetry = PerformanceTelemetry(
            t_ingest=prod_telemetry.t_ingest,
            t_io=prod_telemetry.t_io,
            t_detect=t_detect,
        )

        for detection in detections:
            if detection.confidence < corner_confidence_threshold:
                continue

            # Save the proxy image so that Background Novelty and Tracking can read it.
            # During refinement, this will be overwritten with the high-res frame.
            source_frame_path = (frame_dir / f"video_{video_id}_frame_{row.frame_packet.frame_index}_{row.frame_packet.timestamp_ms}.jpg").resolve()
            if not source_frame_path.exists():
                cv2.imwrite(str(source_frame_path), row.frame_packet.image)

            _put_or_fail(
                detection_queue,
                _DetectionEnvelope(
                    detection_packet=DetectionPacket(
                        frame_index=detection.frame_index,
                        timestamp_ms=detection.timestamp_ms,
                        width=row.frame_packet.width,
                        height=row.frame_packet.height,
                        corner_detection=CornerDetection(
                            corners=detection.polygon,
                            confidence=detection.confidence,
                            metadata=dict(detection.metadata),
                        ),
                        telemetry=telemetry,
                    ),
                    source_frame_path=str(source_frame_path),
                    triage_metrics=row.frame_packet.triage_metrics,
                ),
            )


def _serialize_error(worker: str, exc: Exception) -> dict[str, str]:
    return {
        "worker": worker,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
