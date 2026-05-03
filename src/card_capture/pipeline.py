from __future__ import annotations

import hashlib
import multiprocessing as mp
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full
from typing import List, Optional

import cv2

from .cropper import CardCropper
from .detectors import CardDetector
from .ingestion import FrameTriageFilter, RollingWindowTriage
from .models import (
    CornerDetection,
    DetectionPacket,
    FramePacket,
    FrameSample,
    PerformanceTelemetry,
    ProcessingResult,
    QualityScore,
)
from .scoring import QualityScorer
from .selector import CandidateSelector, HysteresisTracker, ScoredCandidate
from .storage import Storage

_SENTINEL = "__card_capture_queue_sentinel__"
_QUEUE_POLL_INTERVAL_SECONDS = 0.1
_QUEUE_RETRY_MAX_WAIT_SECONDS = 5.0
# Cold model startup and first-batch inference can take well over 30s on a
# fresh environment, especially when weights are downloaded lazily.
_DRAIN_IDLE_TIMEOUT_SECONDS = 300.0


class PipelineTimer:
    def __init__(self):
        self.start_time = time.monotonic()
        self.timings = {}

    def record(self, stage: str):
        self.timings[stage] = time.monotonic() - self.start_time


@dataclass(frozen=True)
class _FrameEnvelope:
    frame_packet: FramePacket
    source_frame_path: str


@dataclass(frozen=True)
class _DetectionEnvelope:
    detection_packet: DetectionPacket
    source_frame_path: str
    triage_metrics: dict[str, float]


@dataclass(frozen=True)
class _ProducerStats:
    frame_count: int
    accepted_frame_count: int


@dataclass(frozen=True)
class ProcessingOptions:
    output_dir: Path
    reader_backend: str = "auto"
    queue_size: int = 64
    inference_batch_size: int = 16
    corner_confidence_threshold: float = 0.5
    blur_threshold: float = 30.0
    variance_threshold: float = 20.0
    empty_pixel_threshold: float = 0.98
    group_gap_ms: int = 300
    spatial_variance_threshold: float = 75.0
    frames_per_instance: int = 2


class VideoProcessor:
    def __init__(
        self,
        storage: Storage,
        sampler,
        detector: CardDetector,
        cropper: Optional[CardCropper] = None,
        scorer: Optional[QualityScorer] = None,
        selector: Optional[CandidateSelector] = None,
    ):
        self.storage = storage
        self.sampler = sampler
        self.detector = detector
        self.cropper = cropper or CardCropper()
        self.scorer = scorer or QualityScorer()
        self.selector = selector

    def process(self, video_path: Path, options: ProcessingOptions) -> ProcessingResult:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video does not exist: {video_path}")

        options.output_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = options.output_dir / "frames"
        best_dir = options.output_dir / "best"
        frame_dir.mkdir(parents=True, exist_ok=True)
        best_dir.mkdir(parents=True, exist_ok=True)

        video_id = self.storage.add_video(
            source_path=str(video_path),
            file_hash=_file_hash(video_path),
            duration_ms=0,
            width=0,
            height=0,
        )
        stats, detection_rows = _run_pipeline_workers(
            video_path=video_path,
            video_id=video_id,
            frame_dir=frame_dir,
            sampler=self.sampler,
            detector=self.detector,
            options=options,
        )

        # Log performance telemetry to the database
        for row in detection_rows:
            if row.detection_packet.telemetry:
                self.storage.add_performance_log(
                    video_id=video_id,
                    frame_index=row.detection_packet.frame_index,
                    telemetry=row.detection_packet.telemetry,
                )

        # v3 adaptive pipeline: Use HysteresisTracker for spatial tracking and flip detection
        candidates = _build_candidates(detection_rows)
        tracker = HysteresisTracker()
        for candidate in candidates:
            tracker.process(candidate)
        
        tracks = tracker.finalize()
        
        # Track which detections are selected as high-quality (canonical)
        selected_detection_ids = set()
        for track in tracks:
            # For each track, select top N frames by quality
            sorted_candidates = sorted(
                track.candidates,
                key=lambda c: c.score.total,
                reverse=True
            )
            for best_c in sorted_candidates[:options.frames_per_instance]:
                selected_detection_ids.add(best_c.detection_id)

        # Persist tracks and views to database
        index_to_view_id: dict[int, int] = {}
        for track in tracks:
            instance_id = self.storage.add_card_instance(
                video_id=video_id,
                track_id=track.instance_id,
            )
            for candidate in track.candidates:
                row = detection_rows[candidate.detection_id]
                is_canonical = candidate.detection_id in selected_detection_ids
                
                confidence = float(row.detection_packet.corner_detection.confidence)
                view_id = self.storage.add_card_view(
                    card_instance_id=instance_id,
                    frame_index=row.detection_packet.frame_index,
                    timestamp_ms=row.detection_packet.timestamp_ms,
                    detection=row.detection_packet.corner_detection,
                    rectified_path=None,
                    quality_score={"confidence": round(confidence, 6)},
                    is_canonical=is_canonical,
                )
                index_to_view_id[candidate.detection_id] = view_id
                
                self.storage.add_evidence_frame(
                    card_view_id=view_id,
                    source_frame_path=row.source_frame_path,
                    frame_width=row.detection_packet.width,
                    frame_height=row.detection_packet.height,
                    metrics=row.triage_metrics,
                )

        # Handle saved_cards (the ones we copy to "best" folder)
        saved_count = 0
        for detection_id in sorted(selected_detection_ids):
            row = detection_rows[detection_id]
            source_path = Path(row.source_frame_path)
            saved_count += 1
            best_path = best_dir / f"video_{video_id}_best_{saved_count}.jpg"
            
            if source_path.exists():
                shutil.copyfile(source_path, best_path)
            
            self.storage.add_saved_card(
                detection_id=index_to_view_id[detection_id],
                image_path=str(best_path),
                final_score=row.detection_packet.corner_detection.confidence,
            )

        self.storage.update_video_status(video_id, "complete" if saved_count > 0 else "no_detections")
        return ProcessingResult(
            video_id=video_id,
            frame_count=stats.frame_count,
            accepted_frame_count=stats.accepted_frame_count,
            detection_count=len(detection_rows),
            saved_instance_count=saved_count,
            output_dir=options.output_dir,
        )


def _build_candidates(rows: list[_DetectionEnvelope]) -> list[ScoredCandidate]:
    candidates: list[ScoredCandidate] = []
    for index, row in enumerate(rows):
        confidence = float(row.detection_packet.corner_detection.confidence)
        score = QualityScore(total=confidence, components={"confidence": round(confidence, 6)})
        # Convert Polygon (4-tuple of Points) to list of (x, y) tuples for spatial clustering
        corners = row.detection_packet.corner_detection.corners
        corner_list = [(float(pt[0]), float(pt[1])) for pt in corners]
        candidates.append(
            ScoredCandidate(
                detection_id=index,
                timestamp_ms=row.detection_packet.timestamp_ms,
                image_path=row.source_frame_path,
                score=score,
                corners=corner_list,
            )
        )
    return candidates


def _run_pipeline_workers(
    video_path: Path,
    video_id: int,
    frame_dir: Path,
    sampler,
    detector,
    options: ProcessingOptions,
) -> tuple[_ProducerStats, list[_DetectionEnvelope]]:
    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=options.queue_size)
    detection_queue = ctx.Queue(maxsize=options.queue_size)
    error_queue = ctx.Queue(maxsize=8)
    stats_queue = ctx.Queue(maxsize=1)
    producer = ctx.Process(
        target=_producer_main,
        args=(
            str(video_path),
            video_id,
            str(frame_dir),
            sampler,
            options.blur_threshold,
            options.variance_threshold,
            options.empty_pixel_threshold,
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
            frame_queue,
            detection_queue,
            error_queue,
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
        stats = _read_producer_stats(stats_queue)
        return stats, detections
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


def _read_producer_stats(stats_queue) -> _ProducerStats:
    try:
        stats = stats_queue.get(timeout=2)
    except Empty as exc:
        raise RuntimeError("producer did not publish frame statistics") from exc
    if isinstance(stats, _ProducerStats):
        return stats
    raise RuntimeError(f"unexpected producer stats payload: {type(stats)!r}")


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


def _producer_main(
    video_path: str,
    video_id: int,
    frame_dir: str,
    sampler,
    blur_threshold: float,
    variance_threshold: float,
    empty_pixel_threshold: float,
    frame_queue,
    stats_queue,
    error_queue,
) -> None:
    frame_count = 0
    accepted_frame_count = 0
    triage = FrameTriageFilter(
        variance_threshold=variance_threshold,
        empty_ratio_threshold=empty_pixel_threshold,
        blur_threshold=blur_threshold,
    )
    adaptive_triage = RollingWindowTriage(window_size=30, keep_percentile=0.5)
    try:
        for frame in sampler.sample(Path(video_path), 0.0):
            timer = PipelineTimer()
            frame_count += 1
            accepted, metrics = triage.evaluate(frame.image)
            timer.record("t_ingest")
            
            if not accepted:
                continue
            
            # Use blur score for rolling window triage to prioritize sharp frames
            if not adaptive_triage.evaluate_score(frame.frame_index, metrics["blur"]):
                continue

            accepted_frame_count += 1
            source_frame_path = Path(frame_dir) / f"video_{video_id}_frame_{frame.frame_index}.jpg"
            cv2.imwrite(str(source_frame_path), frame.image)
            timer.record("t_io")

            t_ingest = timer.timings.get("t_ingest", 0.0)
            t_io = timer.timings.get("t_io", 0.0) - t_ingest

            _put_or_fail(
                frame_queue,
                _FrameEnvelope(
                    frame_packet=FramePacket(
                        frame_index=frame.frame_index,
                        timestamp_ms=frame.timestamp_ms,
                        image=frame.image,
                        width=frame.width,
                        height=frame.height,
                        triage_metrics=metrics,
                        telemetry=PerformanceTelemetry(
                            t_ingest=t_ingest,
                            t_io=t_io,
                        ),
                    ),
                    source_frame_path=str(source_frame_path),
                ),
            )
    except Exception as exc:  # pragma: no cover - exercised in parent integration test.
        _put_with_retry(
            error_queue,
            _serialize_error("producer", exc),
            timeout=_QUEUE_POLL_INTERVAL_SECONDS,
            max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
        )
    finally:
        _put_with_retry(
            stats_queue,
            _ProducerStats(frame_count=frame_count, accepted_frame_count=accepted_frame_count),
            timeout=_QUEUE_POLL_INTERVAL_SECONDS,
            max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
        )
        _put_with_retry(
            frame_queue,
            _SENTINEL,
            timeout=_QUEUE_POLL_INTERVAL_SECONDS,
            max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
        )


def _consumer_main(
    detector,
    inference_batch_size: int,
    corner_confidence_threshold: float,
    frame_queue,
    detection_queue,
    error_queue,
) -> None:
    batch: list[_FrameEnvelope] = []
    try:
        while True:
            item = frame_queue.get()
            if item == _SENTINEL:
                _consume_batch(
                    detector,
                    batch,
                    corner_confidence_threshold,
                    detection_queue,
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
                _consume_batch(
                    detector,
                    batch,
                    corner_confidence_threshold,
                    detection_queue,
                )
                batch = []
    except Exception as exc:  # pragma: no cover - exercised in parent integration test.
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

        source_by_frame = {(row.frame_packet.frame_index, row.frame_packet.timestamp_ms): row for row in batch}
        for detection_packet in detections:
            if detection_packet.corner_detection.confidence < corner_confidence_threshold:
                continue
            source = source_by_frame.get((detection_packet.frame_index, detection_packet.timestamp_ms))
            if source is None:
                continue

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
                    source_frame_path=source.source_frame_path,
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
                    source_frame_path=row.source_frame_path,
                    triage_metrics=row.frame_packet.triage_metrics,
                ),
            )


def _serialize_error(worker: str, exc: Exception) -> dict[str, str]:
    return {
        "worker": worker,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
