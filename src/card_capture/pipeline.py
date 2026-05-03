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

from .cropper import CardCropper, PrecisionNormalizer
from .detectors import CardDetector
from .ingestion import FrameTriageFilter, RollingWindowTriage
from .deduplicator import VisualDeduplicator
from .fuser import MultiFrameFuser
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
        crops_dir = options.output_dir / "crops"
        frame_dir.mkdir(parents=True, exist_ok=True)
        best_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

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

        for row in detection_rows:
            if row.detection_packet.telemetry:
                self.storage.add_performance_log(
                    video_id=video_id,
                    frame_index=row.detection_packet.frame_index,
                    telemetry=row.detection_packet.telemetry,
                )

        candidates = _build_candidates(detection_rows)
        tracker = HysteresisTracker()
        for candidate in candidates:
            tracker.process(candidate)
        
        tracks = tracker.finalize()
        
        normalizer = PrecisionNormalizer()
        deduplicator = VisualDeduplicator()
        fuser = MultiFrameFuser()

        saved_count = 0
        for track in tracks:
            timer = PipelineTimer()
            
            instance_id = self.storage.add_card_instance(
                video_id=video_id,
                track_id=track.instance_id,
                angle=track.angle,
            )

            sorted_candidates = sorted(track.candidates, key=lambda c: c.score.total, reverse=True)
            canonical_candidates = sorted_candidates[:options.frames_per_instance]

            normalized_images = []
            for i, candidate in enumerate(canonical_candidates):
                row = detection_rows[candidate.detection_id]
                raw_image = cv2.imread(row.source_frame_path)
                if raw_image is not None:
                    normalized = normalizer.normalize(raw_image, candidate.corners)
                    normalized_images.append(normalized)
                    
                    rectified_path = crops_dir / f"instance_{instance_id}_view_{i}_rectified.jpg"
                    cv2.imwrite(str(rectified_path), normalized)

            # Global Deduplication
            phash = None
            if normalized_images:
                phash = deduplicator.compute_phash(normalized_images[0])
                duplicate_of = self.storage.find_canonical_for_hash(phash)
                self.storage.update_instance_deduplication(instance_id, phash, duplicate_of)
                
                # Fusion
                fused = fuser.fuse(normalized_images)
                fused_path = best_dir / f"instance_{instance_id}_fused.jpg"
                cv2.imwrite(str(fused_path), fused)
                self.storage.update_instance_fusion(instance_id, str(fused_path))

            timer.record("t_refine")
            t_refine_per_frame = timer.timings["t_refine"] / len(track.candidates) if track.candidates else 0
            
            for candidate in track.candidates:
                row = detection_rows[candidate.detection_id]
                view_id = self.storage.add_card_view(
                    card_instance_id=instance_id,
                    frame_index=row.detection_packet.frame_index,
                    timestamp_ms=row.detection_packet.timestamp_ms,
                    detection=row.detection_packet.corner_detection,
                    glare_x=0.0, glare_y=0.0, sharpness=1.0, # Placeholder metrics
                    is_canonical=candidate in canonical_candidates,
                )
                
                self.storage.add_performance_log(
                    video_id=video_id,
                    frame_index=row.detection_packet.frame_index,
                    telemetry=PerformanceTelemetry(
                        t_ingest=row.detection_packet.telemetry.t_ingest if row.detection_packet.telemetry else 0,
                        t_refine=t_refine_per_frame,
                    )
                )

                self.storage.add_evidence_frame(
                    card_view_id=view_id,
                    source_frame_path=row.source_frame_path,
                    frame_width=row.detection_packet.width,
                    frame_height=row.detection_packet.height,
                    metrics=row.triage_metrics,
                )
            saved_count += 1

        self.storage.update_video_status(video_id, "complete")
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
        corners = row.detection_packet.corner_detection.corners
        candidates.append(
            ScoredCandidate(
                detection_id=index,
                timestamp_ms=row.detection_packet.timestamp_ms,
                image_path=row.source_frame_path,
                score=score,
                corners=[(float(pt[0]), float(pt[1])) for pt in corners],
            )
        )
    return candidates

def _run_pipeline_workers(video_path: Path, video_id: int, frame_dir: Path, sampler, detector, options: ProcessingOptions) -> tuple[_ProducerStats, list[_DetectionEnvelope]]:
    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=options.queue_size)
    detection_queue = ctx.Queue(maxsize=options.queue_size)
    error_queue = ctx.Queue(maxsize=8)
    stats_queue = ctx.Queue(maxsize=1)
    producer = ctx.Process(
        target=_producer_main,
        args=(str(video_path), video_id, str(frame_dir), sampler, options.blur_threshold, options.variance_threshold, options.empty_pixel_threshold, frame_queue, stats_queue, error_queue),
        name="producer",
    )
    consumer = ctx.Process(
        target=_consumer_main,
        args=(detector, options.inference_batch_size, options.corner_confidence_threshold, frame_queue, detection_queue, error_queue),
        name="consumer",
    )
    producer.start()
    consumer.start()
    try:
        detections = _drain_detection_queue(detection_queue, error_queue, producer, consumer, _DRAIN_IDLE_TIMEOUT_SECONDS)
        producer.join(timeout=10)
        consumer.join(timeout=10)
        _raise_child_error_if_present(error_queue, producer, consumer)
        return _read_producer_stats(stats_queue), detections
    finally:
        _stop_process(producer); _stop_process(consumer)
        frame_queue.close(); detection_queue.close(); error_queue.close(); stats_queue.close()

def _read_producer_stats(stats_queue) -> _ProducerStats:
    try: return stats_queue.get(timeout=2)
    except Empty: raise RuntimeError("producer did not publish statistics")

def _drain_detection_queue(detection_queue, error_queue, producer, consumer, idle_timeout_s) -> list[_DetectionEnvelope]:
    rows: list[_DetectionEnvelope] = []
    last_activity = time.monotonic()
    while True:
        try:
            item = detection_queue.get(timeout=_QUEUE_POLL_INTERVAL_SECONDS)
        except Empty:
            if not producer.is_alive() and not consumer.is_alive(): raise RuntimeError("consumer exited before sentinel")
            if time.monotonic() - last_activity > idle_timeout_s: raise RuntimeError("timed out waiting")
            continue
        last_activity = time.monotonic()
        if item == _SENTINEL: return rows
        rows.append(item)

def _raise_child_error_if_present(error_queue, producer, consumer) -> None:
    try: error = error_queue.get_nowait()
    except Empty: return
    _stop_process(producer); _stop_process(consumer)
    raise RuntimeError(str(error))

def _stop_process(process: mp.Process) -> None:
    if process.is_alive(): process.terminate()

def _put_or_fail(q, item) -> None:
    try: q.put(item, timeout=_QUEUE_RETRY_MAX_WAIT_SECONDS)
    except Full: raise RuntimeError("queue full")

def _producer_main(video_path: str, video_id: int, frame_dir: str, sampler, blur_threshold: float, variance_threshold: float, empty_pixel_threshold: float, frame_queue, stats_queue, error_queue) -> None:
    frame_count = 0; accepted_frame_count = 0
    triage = FrameTriageFilter(variance_threshold=variance_threshold, empty_ratio_threshold=empty_pixel_threshold, blur_threshold=blur_threshold)
    adaptive_triage = RollingWindowTriage(window_size=30, keep_percentile=0.5)
    try:
        for frame in sampler.sample(Path(video_path), 0.0):
            timer = PipelineTimer()
            frame_count += 1
            accepted, metrics = triage.evaluate(frame.image)
            timer.record("t_ingest")
            if not accepted or not adaptive_triage.evaluate_score(frame.frame_index, metrics["blur"]): continue
            accepted_frame_count += 1
            source_frame_path = Path(frame_dir) / f"video_{video_id}_frame_{frame.frame_index}.jpg"
            cv2.imwrite(str(source_frame_path), frame.image)
            timer.record("t_io")
            _put_or_fail(frame_queue, _FrameEnvelope(frame_packet=FramePacket(
                frame_index=frame.frame_index, timestamp_ms=frame.timestamp_ms, image=frame.image, width=frame.width, height=frame.height,
                triage_metrics=metrics, telemetry=PerformanceTelemetry(t_ingest=timer.timings.get("t_ingest", 0.0), t_io=timer.timings.get("t_io", 0.0))
            ), source_frame_path=str(source_frame_path)))
    finally:
        _put_or_fail(stats_queue, _ProducerStats(frame_count=frame_count, accepted_frame_count=accepted_frame_count))
        _put_or_fail(frame_queue, _SENTINEL)

def _consumer_main(detector, inference_batch_size: int, corner_confidence_threshold: float, frame_queue, detection_queue, error_queue) -> None:
    batch: list[_FrameEnvelope] = []
    while True:
        item = frame_queue.get()
        if item == _SENTINEL:
            _consume_batch(detector, batch, corner_confidence_threshold, detection_queue)
            _put_or_fail(detection_queue, _SENTINEL)
            return
        batch.append(item)
        if len(batch) >= inference_batch_size:
            _consume_batch(detector, batch, corner_confidence_threshold, detection_queue)
            batch = []

def _consume_batch(detector, batch, corner_confidence_threshold, detection_queue) -> None:
    if not batch: return
    timer = PipelineTimer()
    detections = detector.detect_batch([row.frame_packet for row in batch], confidence_threshold=corner_confidence_threshold)
    timer.record("t_detect")
    for detection_packet in detections:
        _put_or_fail(detection_queue, _DetectionEnvelope(
            detection_packet=detection_packet,
            source_frame_path=next(r.source_frame_path for r in batch if r.frame_packet.frame_index == detection_packet.frame_index),
            triage_metrics=next(r.frame_packet.triage_metrics for r in batch if r.frame_packet.frame_index == detection_packet.frame_index)
        ))

def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
