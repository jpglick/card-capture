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
from .ingestion import FrameTriageFilter
from .models import (
    CornerDetection,
    DetectionPacket,
    FramePacket,
    FrameSample,
    ProcessingResult,
    QualityScore,
)
from .scoring import QualityScorer
from .selector import CandidateSelector, ScoredCandidate
from .storage import Storage

_SENTINEL = "__card_capture_queue_sentinel__"


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

        selector = self.selector or CandidateSelector()
        candidates = _build_candidates(detection_rows)
        selected = selector.select(candidates)
        selected_ids = {candidate.detection_id for candidate in selected}

        for row_index, row in enumerate(detection_rows):
            instance_id = self.storage.add_card_instance(
                video_id=video_id,
                track_id=f"card_{row_index + 1}",
            )
            confidence = float(row.detection_packet.corner_detection.confidence)
            view_id = self.storage.add_card_view(
                card_instance_id=instance_id,
                frame_index=row.detection_packet.frame_index,
                timestamp_ms=row.detection_packet.timestamp_ms,
                detection=row.detection_packet.corner_detection,
                rectified_path=None,
                quality_score={"confidence": round(confidence, 6)},
                is_canonical=row_index in selected_ids,
            )
            self.storage.add_evidence_frame(
                card_view_id=view_id,
                source_frame_path=row.source_frame_path,
                frame_width=row.detection_packet.width,
                frame_height=row.detection_packet.height,
                metrics=row.triage_metrics,
            )

        for selected_index, candidate in enumerate(selected):
            source_path = Path(candidate.image_path)
            best_path = best_dir / f"video_{video_id}_best_{selected_index + 1}.jpg"
            if source_path.exists():
                shutil.copyfile(source_path, best_path)

        self.storage.update_video_status(video_id, "complete" if selected else "no_detections")
        return ProcessingResult(
            video_id=video_id,
            frame_count=stats.frame_count,
            accepted_frame_count=stats.accepted_frame_count,
            detection_count=len(detection_rows),
            saved_instance_count=len(selected),
            output_dir=options.output_dir,
        )


def _build_candidates(rows: list[_DetectionEnvelope]) -> list[ScoredCandidate]:
    candidates: list[ScoredCandidate] = []
    for index, row in enumerate(rows):
        confidence = float(row.detection_packet.corner_detection.confidence)
        score = QualityScore(total=confidence, components={"confidence": round(confidence, 6)})
        candidates.append(
            ScoredCandidate(
                detection_id=index,
                timestamp_ms=row.detection_packet.timestamp_ms,
                image_path=row.source_frame_path,
                score=score,
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
        detections = _drain_detection_queue(detection_queue, error_queue, producer, consumer)
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


def _drain_detection_queue(detection_queue, error_queue, producer, consumer) -> list[_DetectionEnvelope]:
    rows: list[_DetectionEnvelope] = []
    while True:
        _raise_child_error_if_present(error_queue, producer, consumer)
        try:
            item = detection_queue.get(timeout=0.1)
        except Empty:
            if not producer.is_alive() and not consumer.is_alive():
                _raise_child_error_if_present(error_queue, producer, consumer)
                raise RuntimeError("consumer exited before emitting sentinel")
            continue
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
    if process.exitcode not in (0, None):
        raise RuntimeError(f"{process.name} exited with code {process.exitcode}")


def _stop_process(process: mp.Process) -> None:
    if process.is_alive():
        process.terminate()


def _put_with_retry(q, item, timeout: float = 0.1) -> None:
    backoff = 0.01
    while True:
        try:
            q.put(item, timeout=timeout)
            return
        except Full:
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 0.2)


def _safe_put(q, item) -> None:
    _put_with_retry(q, item, timeout=0.1)


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
    try:
        for frame in sampler.sample(Path(video_path), 0.0):
            frame_count += 1
            accepted, metrics = triage.evaluate(frame.image)
            if not accepted:
                continue
            accepted_frame_count += 1
            source_frame_path = Path(frame_dir) / f"video_{video_id}_frame_{frame.frame_index}.jpg"
            cv2.imwrite(str(source_frame_path), frame.image)
            _safe_put(
                frame_queue,
                _FrameEnvelope(
                    frame_packet=FramePacket(
                        frame_index=frame.frame_index,
                        timestamp_ms=frame.timestamp_ms,
                        image=frame.image,
                        width=frame.width,
                        height=frame.height,
                        triage_metrics=metrics,
                    ),
                    source_frame_path=str(source_frame_path),
                ),
            )
    except Exception as exc:  # pragma: no cover - exercised in parent integration test.
        _put_with_retry(error_queue, _serialize_error("producer", exc), timeout=0.1)
    finally:
        _put_with_retry(
            stats_queue,
            _ProducerStats(frame_count=frame_count, accepted_frame_count=accepted_frame_count),
            timeout=0.1,
        )
        _put_with_retry(frame_queue, _SENTINEL, timeout=0.1)


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
                _put_with_retry(detection_queue, _SENTINEL, timeout=0.1)
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
        _put_with_retry(error_queue, _serialize_error("consumer", exc), timeout=0.1)
        _put_with_retry(detection_queue, _SENTINEL, timeout=0.1)


def _consume_batch(
    detector,
    batch: list[_FrameEnvelope],
    corner_confidence_threshold: float,
    detection_queue,
) -> None:
    if not batch:
        return

    if hasattr(detector, "detect_batch"):
        frames = [row.frame_packet for row in batch]
        detections = detector.detect_batch(frames, confidence_threshold=corner_confidence_threshold)
        source_by_frame = {(row.frame_packet.frame_index, row.frame_packet.timestamp_ms): row for row in batch}
        for detection_packet in detections:
            if detection_packet.corner_detection.confidence < corner_confidence_threshold:
                continue
            source = source_by_frame.get((detection_packet.frame_index, detection_packet.timestamp_ms))
            if source is None:
                continue
            _safe_put(
                detection_queue,
                _DetectionEnvelope(
                    detection_packet=detection_packet,
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
        for detection in detector.detect(legacy_frame):
            if detection.confidence < corner_confidence_threshold:
                continue
            _safe_put(
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
