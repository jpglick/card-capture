from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full
from typing import List, Optional, Any

import cv2
import numpy as np

from .cropper import CardCropper, PrecisionNormalizer
from .gpu_refinement import KorniaNormalizer
from .detectors import CardDetector
from .deduplicator import VisualDeduplicator
from .fuser import calculate_sharpness, find_glare_centroid
from .ingestion import FrameTriageFilter
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
_QUEUE_RETRY_MAX_WAIT_SECONDS = 60.0
_DRAIN_IDLE_TIMEOUT_SECONDS = 300.0
_CANONICAL_TARGET_FRAMES = 3
_CANONICAL_MAX_FRAMES = 4
_SAME_APPEARANCE_HAMMING_MAX = 8
_SESSION_DUPLICATE_HAMMING_MAX = 6
_SESSION_TEXTINESS_MARGIN = 0.03
_SESSION_APPEARANCE_SIMILARITY_MIN = 0.995
_SESSION_MERGE_SIMILARITY_MIN = 0.99


class PipelineTimer:
    def __init__(self):
        self.start_time = time.monotonic()
        self.timings: dict[str, float] = {}

    def record(self, stage: str):
        self.timings[stage] = time.monotonic() - self.start_time


@dataclass
class _PreparedTrack:
    track: Any
    session_id: int
    first_frame_index: int
    angle: str
    frame_entries: list[dict]
    canonical_entries: list[dict]
    candidate_hashes: list[str]
    primary_hash: str
    side_score: float
    appearance_vector: np.ndarray
    canonical_detection_ids: set[int]
    duplicate_track_index: Optional[int] = None


@dataclass(frozen=True)
class _FrameEnvelope:
    frame_packet: FramePacket


@dataclass(frozen=True)
class _DetectionEnvelope:
    detection_packet: DetectionPacket
    source_frame_path: str
    image_jpeg: bytes
    triage_metrics: dict[str, float]


@dataclass(frozen=True)
class _ProducerStats:
    frame_count: int
    accepted_frame_count: int
    accepted_frame_presence: list[tuple[int, int, bool]] = field(default_factory=list)
    sampler_telemetry: dict[str, Any] = field(default_factory=dict)


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
    rotate_180: bool = True




class NullStateDetector:
    def __init__(self, frames: int = 30, threshold: float = 15.0):
        self.frames = frames
        self.threshold = threshold
        self.background_model = None
        self.frame_count = 0

    def is_workspace_empty(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.background_model is None:
            self.background_model = np.zeros_like(gray, dtype=np.float32)

        if self.frame_count < self.frames:
            self.background_model = (
                (self.background_model * self.frame_count + gray) / (self.frame_count + 1)
            )
            self.frame_count += 1
            return False # Return False during warmup so we don't accidentally trigger resets
        
        diff = cv2.absdiff(gray, self.background_model.astype(np.uint8))
        return float(np.mean(diff)) < self.threshold


class SessionManager:
    def __init__(self):
        self.active_session_id: Optional[str] = None
        
    def start_session(self, timestamp: int):
        self.active_session_id = str(timestamp)

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
        self.normalizer = PrecisionNormalizer()
        self.kornia_normalizer: Optional[KorniaNormalizer] = None
        self.scorer = scorer or QualityScorer()
        self.selector = selector
        self.null_detector = None
        self.session_manager = SessionManager()
        self.tracker = HysteresisTracker(max_dist=150.0, min_track_length=12, max_gap_frames=15)

    def flush_tracker(self):
        self.tracker.finalize()
        self.tracker.reset_active()

    def process(self, video_path: Path, options: ProcessingOptions, debug_config: Any = None) -> ProcessingResult:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video does not exist: {video_path}")
            
        self.null_detector = NullStateDetector(frames=options.background_frames, threshold=options.background_threshold)
        if options.use_kornia:
            try:
                self.kornia_normalizer = KorniaNormalizer(
                    width=self.normalizer.width,
                    height=self.normalizer.height,
                    device=options.kornia_device,
                )
            except Exception:
                self.kornia_normalizer = None


        output_dir = options.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = output_dir / "frames"
        crops_dir = output_dir / "crops"
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

        candidates = _build_candidates(detection_rows)
        candidate_confidences = [candidate.score.total for candidate in candidates]
        if candidate_confidences:
            tracker_t_high = float(
                np.clip(np.percentile(candidate_confidences, 65), 0.40, 0.75)
            )
        else:
            tracker_t_high = 0.60
        tracker_t_low = max(0.20, tracker_t_high - 0.20)
        adaptive_min_track_length = max(
            3,
            min(options.min_track_length, max(3, len(detection_rows) // 3)),
        )
        self.tracker = HysteresisTracker(
            t_high=tracker_t_high,
            t_low=tracker_t_low,
            max_dist=options.spatial_variance_threshold,
            min_track_length=adaptive_min_track_length,
            max_gap_frames=options.null_patience_frames * 2, # Allow for slightly larger gaps in sparse sampling
        )
        by_frame: dict[int, list[ScoredCandidate]] = {}
        for candidate in candidates:
            key = -1 if candidate.frame_index is None else int(candidate.frame_index)
            by_frame.setdefault(key, []).append(candidate)

        current_session_id = 0
        frame_to_session: dict[int, int] = {}
        last_frame_idx = -1
        
        # We rely purely on the AdaptivePresenceSampler's temporal gaps to define sessions.
        # The sampler omits frames where the workspace is empty.
        # If the gap between two accepted frames is large, a physical swap occurred.
        for frame_index, timestamp_ms, _ in stats.accepted_frame_presence:
            if last_frame_idx != -1 and (frame_index - last_frame_idx) > options.null_patience_frames:
                self.tracker.finalize()
                self.tracker.record_reset_event(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    reason="sampled_frame_gap",
                    gap_frames=frame_index - last_frame_idx,
                )
                print(f"[Stage: Tracking] | Session: {current_session_id} | Action: Session Reset (Gap: {frame_index - last_frame_idx} frames)")
                self.session_manager.active_session_id = None
                self.tracker.reset_active()
            last_frame_idx = frame_index

            frame_candidates = by_frame.get(frame_index, [])

            if self.session_manager.active_session_id is None:
                self.session_manager.active_session_id = str(timestamp_ms)
                current_session_id += 1
            frame_to_session[frame_index] = current_session_id
            self.tracker.tick()
            for candidate in frame_candidates:
                self.tracker.process(candidate)

        raw_track_lengths = [len(track.candidates) for track in self.tracker.active_tracks]
        tracker_events = list(self.tracker.association_events)
        tracks = self.tracker.finalize()

        normalizer = PrecisionNormalizer()
        deduplicator = VisualDeduplicator()
        prepared_tracks: list[_PreparedTrack] = []
        saved_count = 0
        for track in tracks:
            # Sort by total score to find best candidates, but cap at top 8 to prevent massive CPU/GPU burn
            scored_track = sorted(track.candidates, key=lambda c: c.score.total, reverse=True)[:8]
            
            frame_entries = []
            normalized_by_detection: dict[int, np.ndarray] = {}
            raw_images_by_detection: dict[int, np.ndarray] = {}
            
            if self.kornia_normalizer is not None:
                batch_items: list[tuple[np.ndarray, list[tuple[float, float]]]] = []
                batch_ids: list[int] = []
                for candidate in scored_track:
                    if not candidate.corners:
                        continue
                    row = detection_rows[candidate.detection_id]
                    raw_image = _decode_detection_image(row)
                    if raw_image is None:
                        continue
                    raw_images_by_detection[candidate.detection_id] = raw_image
                    batch_items.append((raw_image, candidate.corners))
                    batch_ids.append(candidate.detection_id)
                if batch_items:
                    try:
                        warped = self.kornia_normalizer.warp_canonical_batch(batch_items)
                        for detection_id, image in zip(batch_ids, warped):
                            normalized_by_detection[detection_id] = image
                    except Exception as e:
                        print(f"Kornia warp failed: {e}")
                        normalized_by_detection = {}
                        
            for candidate in scored_track:
                row = detection_rows[candidate.detection_id]
                raw_image = raw_images_by_detection.get(candidate.detection_id)
                if raw_image is None:
                    raw_image = _decode_detection_image(row)
                    if raw_image is None:
                        continue

                from .selector import _get_polygon_area, _aspect_ratio
                if candidate.corners:
                    area = _get_polygon_area(candidate.corners)
                    aspect = _aspect_ratio(candidate.corners)
                    cx = sum(p[0] for p in candidate.corners) / 4.0
                    cy = sum(p[1] for p in candidate.corners) / 4.0
                    self.storage.add_track_telemetry(
                        video_id, track.instance_id, candidate.frame_index, area, aspect, cx, cy
                    )

                if debug_config and debug_config.export_frames:
                    debug_dir = output_dir / "debug_frames"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    debug_img = raw_image.copy()
                    
                    if candidate.corners:
                        pts = np.array(candidate.corners, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(debug_img, [pts], True, (0, 0, 255), 2)
                        
                        for i, pt in enumerate(candidate.corners):
                            cv2.putText(debug_img, str(i), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.putText(debug_img, f"Track: {track.instance_id[:8]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    out_path = debug_dir / f"frame_{candidate.frame_index}_track_{track.instance_id[:8]}.jpg"
                    cv2.imwrite(str(out_path), debug_img)

                normalized = normalized_by_detection.get(candidate.detection_id)
                if normalized is None:
                    normalized = normalizer.normalize(raw_image, candidate.corners)
                glare_centroid = find_glare_centroid(normalized)
                glare_x, glare_y = glare_centroid if glare_centroid else (None, None)
                frame_entries.append(
                    {
                        "candidate": candidate,
                        "row": row,
                        "normalized": normalized,
                        "visual_hash": deduplicator.compute_phash(normalized),
                        "glare_x": glare_x,
                        "glare_y": glare_y,
                        "sharpness": calculate_sharpness(normalized),
                        "glare_mask": _compress_array(_glare_mask(normalized)),
                        "laplacian_heatmap": _compress_array(_laplacian_heatmap(normalized)),
                    }
                )

            if not frame_entries:
                continue

            canonical_entries = _select_canonical_entries(frame_entries, deduplicator)
            canonical_detection_ids = {
                entry["candidate"].detection_id for entry in canonical_entries
            }
            best_canonical = max(canonical_entries, key=lambda e: e["candidate"].score.total)
            phash = best_canonical["visual_hash"]
            candidate_hashes: list[str] = []
            for entry in canonical_entries:
                h = str(entry["visual_hash"])
                candidate_hashes.append(h)
                rotated = cv2.rotate(entry["normalized"], cv2.ROTATE_180)
                candidate_hashes.append(deduplicator.compute_phash(rotated))
            first_frame_index = (
                -1 if track.candidates[0].frame_index is None else int(track.candidates[0].frame_index)
            )
            prepared_tracks.append(
                _PreparedTrack(
                    track=track,
                    session_id=frame_to_session.get(first_frame_index, 0),
                    first_frame_index=first_frame_index,
                    angle=track.angle,
                    frame_entries=frame_entries,
                    canonical_entries=canonical_entries,
                    candidate_hashes=candidate_hashes,
                    primary_hash=str(phash),
                    side_score=_side_textiness_score(best_canonical["normalized"]),
                    appearance_vector=_appearance_vector(best_canonical["normalized"]),
                    canonical_detection_ids=canonical_detection_ids,
                )
            )

        _resolve_session_tracks(prepared_tracks, deduplicator)
        duplicate_track_count = sum(
            1 for prepared in prepared_tracks if prepared.duplicate_track_index is not None
        )
        track_lengths = [len(track.candidates) for track in tracks]
        sampler_telemetry = dict(stats.sampler_telemetry)
        if candidate_confidences:
            sampler_telemetry["candidate_confidence_min"] = float(min(candidate_confidences))
            sampler_telemetry["candidate_confidence_p50"] = float(
                np.percentile(candidate_confidences, 50)
            )
            sampler_telemetry["candidate_confidence_p90"] = float(
                np.percentile(candidate_confidences, 90)
            )
        sampler_telemetry["tracker_t_high"] = tracker_t_high
        sampler_telemetry["tracker_t_low"] = tracker_t_low
        sampler_telemetry["adaptive_min_track_length"] = adaptive_min_track_length
        sampler_telemetry["detections"] = len(detection_rows)
        sampler_telemetry["raw_track_lengths"] = raw_track_lengths
        sampler_telemetry["tracks_finalized"] = len(tracks)
        sampler_telemetry["track_lengths"] = track_lengths
        sampler_telemetry["duplicate_tracks"] = duplicate_track_count
        sampler_telemetry["tracker_event_count"] = len(tracker_events)
        sampler_telemetry["tracker_event_actions"] = _count_event_values(tracker_events, "action")
        sampler_telemetry["tracker_split_reasons"] = _count_event_values(tracker_events, "split_reason")

        track_instance_ids: dict[int, int] = {}
        for track_index, prepared in enumerate(prepared_tracks):
            timer = PipelineTimer()
            instance_id = self.storage.add_card_instance(
                video_id=video_id,
                track_id=prepared.track.instance_id,
                angle=prepared.angle,
            )
            track_instance_ids[track_index] = instance_id

            duplicate_of = None
            if prepared.duplicate_track_index is not None:
                duplicate_of = track_instance_ids.get(prepared.duplicate_track_index)
            if duplicate_of is None:
                duplicate_of = self.storage.find_canonical_for_hashes(
                    prepared.candidate_hashes,
                    threshold=6,
                )
            self.storage.update_instance_deduplication(instance_id, prepared.primary_hash, duplicate_of)

            rectified_paths: dict[int, str] = {}
            for canonical_order, entry in enumerate(prepared.canonical_entries):
                candidate = entry["candidate"]
                rectified_path = (
                    crops_dir / f"instance_{instance_id}_view_{canonical_order}_rectified.jpg"
                ).resolve()
                cv2.imwrite(str(rectified_path), entry["normalized"])
                rectified_paths[candidate.detection_id] = str(rectified_path)

            timer.record("t_refine")
            t_refine_per_frame = timer.timings["t_refine"] / max(1, len(prepared.track.candidates))

            view_ids_by_detection: dict[int, int] = {}
            persisted_frames: dict[tuple[int, int], str] = {}
            for entry in prepared.frame_entries:
                candidate = entry["candidate"]
                row = entry["row"]
                is_canonical = candidate.detection_id in prepared.canonical_detection_ids
                view_id = self.storage.add_card_view(
                    card_instance_id=instance_id,
                    frame_index=row.detection_packet.frame_index,
                    timestamp_ms=row.detection_packet.timestamp_ms,
                    detection=row.detection_packet.corner_detection,
                    rectified_path=rectified_paths.get(candidate.detection_id),
                    quality_score={"confidence": round(row.detection_packet.corner_detection.confidence, 6)},
                    is_canonical=is_canonical,
                    glare_x=entry["glare_x"],
                    glare_y=entry["glare_y"],
                    sharpness=entry["sharpness"],
                    glare_mask=(
                        entry["glare_mask"]
                        if options.telemetry_scope == "all" or is_canonical
                        else None
                    ),
                    laplacian_heatmap=(
                        entry["laplacian_heatmap"]
                        if options.telemetry_scope == "all" or is_canonical
                        else None
                    ),
                    initial_confidence=float(row.detection_packet.corner_detection.confidence),
                )
                view_ids_by_detection[candidate.detection_id] = view_id

                if options.telemetry_scope == "all" or is_canonical:
                    telemetry = row.detection_packet.telemetry or PerformanceTelemetry()
                    self.storage.add_performance_log(
                        video_id=video_id,
                        frame_index=row.detection_packet.frame_index,
                        telemetry=PerformanceTelemetry(
                            t_ingest=telemetry.t_ingest,
                            t_detect=telemetry.t_detect,
                            t_refine=t_refine_per_frame,
                            t_io=telemetry.t_io,
                            queue_wait=telemetry.queue_wait,
                        ),
                    )
                self.storage.add_evidence_frame(
                    card_view_id=view_id,
                    source_frame_path=_persist_source_frame(
                        frame_dir=frame_dir,
                        video_id=video_id,
                        row=row,
                        persisted_frames=persisted_frames,
                    ),
                    frame_width=row.detection_packet.width,
                    frame_height=row.detection_packet.height,
                    metrics=dict(row.triage_metrics),
                )

            best_detection_id = prepared.canonical_entries[0]["candidate"].detection_id
            best_view_id = view_ids_by_detection.get(best_detection_id)
            if best_view_id is not None:
                canonical_image_path = rectified_paths.get(best_detection_id)
                if canonical_image_path is None:
                    continue
                if duplicate_of is None:
                    self.storage.add_saved_card(
                        detection_id=best_view_id,
                        image_path=canonical_image_path,
                        final_score=float(prepared.canonical_entries[0]["candidate"].score.total),
                    )
                    saved_count += 1

        if saved_count > 0:
            status = "complete"
        elif detection_rows and tracks:
            status = "no_saves"
        elif detection_rows:
            status = "no_tracks"
        else:
            status = "no_detections"
        self.storage.update_video_status(video_id, status)

        telemetry_path = output_dir / "run_telemetry.json"
        tracker_events_path = output_dir / "tracker_association_events.json"
        tracker_events_path.write_text(json.dumps(tracker_events, indent=2, sort_keys=True))
        sampler_telemetry["tracker_association_events_path"] = str(tracker_events_path)
        sampler_telemetry["status"] = status
        sampler_telemetry["saved_instances"] = saved_count
        sampler_telemetry["accepted_frames"] = stats.accepted_frame_count
        sampler_telemetry["frame_count"] = stats.frame_count
        telemetry_path.write_text(json.dumps(sampler_telemetry, indent=2, sort_keys=True))
        return ProcessingResult(
            video_id=video_id,
            frame_count=stats.frame_count,
            accepted_frame_count=stats.accepted_frame_count,
            detection_count=len(detection_rows),
            saved_instance_count=saved_count,
            output_dir=output_dir,
            telemetry={**sampler_telemetry, "telemetry_path": str(telemetry_path)},
        )


def _count_event_values(events: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        value = event.get(key)
        if value is None:
            continue
        value_key = str(value)
        counts[value_key] = counts.get(value_key, 0) + 1
    return counts


def _side_textiness_score(image: np.ndarray) -> float:
    height, width = image.shape[:2]
    margin_h = int(height * 0.15)
    margin_w = int(width * 0.15)
    inner = image[margin_h : height - margin_h, margin_w : width - margin_w]
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(edges.mean() / 255.0)
    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    ink_ratio = float(thresholded.mean() / 255.0)
    return edge_ratio + ink_ratio


def _appearance_vector(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    margin_h = int(height * 0.15)
    margin_w = int(width * 0.15)
    inner = image[margin_h : height - margin_h, margin_w : width - margin_w]
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    small_std = float(small.std())
    if small_std > 1e-6:
        small /= small_std
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).astype(np.float32)
    hist = hist.flatten()
    hist_sum = float(hist.sum())
    if hist_sum > 1e-6:
        hist /= hist_sum
    vector = np.concatenate([small.flatten(), hist])
    norm = float(np.linalg.norm(vector))
    if norm > 1e-6:
        vector /= norm
    return vector


def _appearance_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b))


def _persist_source_frame(
    frame_dir: Path,
    video_id: int,
    row: _DetectionEnvelope,
    persisted_frames: dict[tuple[int, int], str],
) -> str:
    key = (int(row.detection_packet.frame_index), int(row.detection_packet.timestamp_ms))
    existing = persisted_frames.get(key)
    if existing is not None:
        return existing
    frame_dir.mkdir(parents=True, exist_ok=True)
    path = (
        frame_dir
        / f"video_{video_id}_frame_{row.detection_packet.frame_index}_{row.detection_packet.timestamp_ms}.jpg"
    ).resolve()
    path.write_bytes(row.image_jpeg)
    persisted = str(path)
    persisted_frames[key] = persisted
    return persisted


def _decode_detection_image(row: _DetectionEnvelope) -> Optional[np.ndarray]:
    array = np.frombuffer(row.image_jpeg, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _min_hash_distance(
    hashes_a: list[str],
    hashes_b: list[str],
    deduplicator: VisualDeduplicator,
) -> int:
    return min(
        deduplicator.hamming_distance(hash_a, hash_b)
        for hash_a in hashes_a
        for hash_b in hashes_b
    )


def _resolve_session_tracks(
    prepared_tracks: list[_PreparedTrack],
    deduplicator: VisualDeduplicator,
) -> None:
    by_session: dict[int, list[tuple[int, _PreparedTrack]]] = {}
    for track_index, prepared in enumerate(prepared_tracks):
        by_session.setdefault(prepared.session_id, []).append((track_index, prepared))

    for session_tracks in by_session.values():
        if not session_tracks:
            continue
            
        # Sort by track length (descending)
        session_tracks.sort(key=lambda item: len(item[1].track.candidates), reverse=True)
        
        # Longest track is front
        first_index, first_prepared = session_tracks[0]
        first_prepared.duplicate_track_index = None
        first_prepared.angle = "Front"
        
        # Second longest is back
        if len(session_tracks) > 1:
            second_index, second_prepared = session_tracks[1]
            second_prepared.duplicate_track_index = first_index
            second_prepared.angle = "Back"
            
            # Any remaining are fragments, merge them to the front
            for frag_index, frag_prepared in session_tracks[2:]:
                frag_prepared.duplicate_track_index = first_index
                frag_prepared.angle = "Front"

def _build_candidates(rows: list[_DetectionEnvelope]) -> list[ScoredCandidate]:
    candidates: list[ScoredCandidate] = []
    for index, row in enumerate(rows):
        confidence = float(row.detection_packet.corner_detection.confidence)
        score = QualityScore(total=confidence, components={"confidence": round(confidence, 6)})
        corners = row.detection_packet.corner_detection.corners
        corner_list = [(float(pt[0]), float(pt[1])) for pt in corners]
        candidates.append(
            ScoredCandidate(
                detection_id=index,
                timestamp_ms=row.detection_packet.timestamp_ms,
                image_path=row.source_frame_path,
                score=score,
                corners=corner_list,
                frame_index=row.detection_packet.frame_index,
            )
        )
    return candidates


def _select_canonical_entries(frame_entries: list[dict], deduplicator: VisualDeduplicator) -> list[dict]:
    if not frame_entries:
        return []

    scored = sorted(
        frame_entries,
        key=lambda entry: (
            float(entry["candidate"].score.total),
            float(entry["sharpness"]),
        ),
        reverse=True,
    )
    anchor = scored[0]
    anchor_hash = str(anchor["visual_hash"])

    for entry in frame_entries:
        entry["_hamming_to_anchor"] = deduplicator.hamming_distance(
            str(entry["visual_hash"]), anchor_hash
        )

    same_appearance = [
        entry
        for entry in frame_entries
        if int(entry["_hamming_to_anchor"]) <= _SAME_APPEARANCE_HAMMING_MAX
    ]

    target = min(_CANONICAL_TARGET_FRAMES, len(frame_entries))
    if len(same_appearance) < target:
        same_appearance = sorted(
            frame_entries,
            key=lambda entry: (
                int(entry["_hamming_to_anchor"]),
                -float(entry["candidate"].score.total),
                -float(entry["sharpness"]),
            ),
        )[: min(_CANONICAL_MAX_FRAMES, len(frame_entries))]

    ranked = sorted(
        same_appearance,
        key=lambda entry: (
            float(entry["candidate"].score.total),
            float(entry["sharpness"]),
        ),
        reverse=True,
    )

    selected: list[dict] = [ranked[0]]
    while len(selected) < min(target, len(ranked)):
        best_entry = None
        best_key = None
        for entry in ranked:
            if entry in selected:
                continue
            min_gap = min(
                abs(
                    int(entry["candidate"].timestamp_ms)
                    - int(prev["candidate"].timestamp_ms)
                )
                for prev in selected
            )
            key = (
                min_gap,
                float(entry["candidate"].score.total),
                float(entry["sharpness"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_entry = entry
        if best_entry is None:
            break
        selected.append(best_entry)

    return selected


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
            sampler,
            options.blur_threshold,
            options.variance_threshold,
            options.empty_pixel_threshold,
            options.background_frames,
            options.background_threshold,
            options.triage_keep_percentile,
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
    sampler,
    blur_threshold: float,
    variance_threshold: float,
    empty_pixel_threshold: float,
    background_frames: int,
    background_threshold: float,
    triage_keep_percentile: float,
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
    null_detector = NullStateDetector(frames=background_frames, threshold=background_threshold)
    try:
        for frame in sampler.sample(Path(video_path), 0.0):
            timer = PipelineTimer()
            frame_count += 1
            accepted, metrics = triage.evaluate(frame.image)
            timer.record("t_ingest")

            if not accepted:
                continue

            accepted_frame_count += 1
            accepted_frame_presence.append((frame.frame_index, frame.timestamp_ms, False))
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
    except Exception as exc:
        _put_with_retry(
            error_queue,
            _serialize_error("producer", exc),
            timeout=_QUEUE_POLL_INTERVAL_SECONDS,
            max_wait_s=_QUEUE_RETRY_MAX_WAIT_SECONDS,
        )
    finally:
        sampler_telemetry: dict[str, Any] = {"sampler_type": type(sampler).__name__}
        for attr in (
            "last_scan_frame_count",
            "last_presence_window_count",
            "last_selected_frame_count",
            "last_score_threshold",
            "last_fallback_used",
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
            encoded, image_jpeg = cv2.imencode(".jpg", source.frame_packet.image)
            if not encoded:
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
                    source_frame_path=(
                        f"video_frame_{source.frame_packet.frame_index}_{source.frame_packet.timestamp_ms}.jpg"
                    ),
                    image_jpeg=image_jpeg.tobytes(),
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
            encoded, image_jpeg = cv2.imencode(".jpg", row.frame_packet.image)
            if not encoded:
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
                    source_frame_path=(
                        f"video_frame_{row.frame_packet.frame_index}_{row.frame_packet.timestamp_ms}.jpg"
                    ),
                    image_jpeg=image_jpeg.tobytes(),
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


def _glare_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def _laplacian_heatmap(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return lap.astype(np.float32)


def _compress_array(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=array)
    return buffer.getvalue()
