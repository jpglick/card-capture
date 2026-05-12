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
from .fuser import calculate_sharpness, find_glare_centroid, MultiFrameFuser
from .ingestion import FrameTriageFilter, _open_capture
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
from .selector import CandidateSelector, ScoredCandidate
from .tracking import ByteTrackAdapter
from .storage import Storage
from .adaptive_gap import compute_session_gap_frames
from .presence.background_novelty import BackgroundModel
from .analysis.hard_case_capture import is_hard_case, capture_hard_case
from .calibration.per_video_adaptive import AdaptiveThresholdComputer

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
_SAME_CARD_HAMMING_MAX = 22  # 16-hex-char (64-bit) phash; 22/64 ≈ 34% bits differ


def adaptive_min_track_length(
    detection_count: int,
    inter_gap_frames: list[float],
    min_baseline: int = 3,
) -> int:
    """Compute adaptive min_track_length from inter-detection gaps.

    Replaces the formula: len(detection_rows) // 3 (which is too high
    for long single-card videos) with:
    max(min_baseline, median_gap × 3)

    This scales the threshold to the typical card-swap frequency rather
    than the absolute detection count.

    Args:
        detection_count: Total number of detections (unused in new formula, kept for API)
        inter_gap_frames: List of inter-detection gap frame counts
        min_baseline: Minimum threshold (default 3)

    Returns:
        max(min_baseline, int(median_gap × 3))

    Example:
        Long single-card video (300 detections, gaps ~2 frames):
        - Old: 300 // 3 = 100 (too high)
        - New: max(3, 2 × 3) = 6 (better)

        Frequent swaps (300 detections, gaps ~11 frames):
        - Old: 300 // 3 = 100
        - New: max(3, 11 × 3) = 33 (appropriate for swap frequency)
    """
    if not inter_gap_frames:
        return min_baseline

    median_gap = float(np.median(inter_gap_frames))
    return max(min_baseline, int(median_gap * 3))


class PipelineTimer:
    def __init__(self):
        self.start_time = time.monotonic()
        self.timings: dict[str, float] = {}

    def record(self, stage: str):
        self.timings[stage] = time.monotonic() - self.start_time


@dataclass
class _PreparedTrack:
    """Prepared track with canonical entries and foil-aware fusion.

    Attributes:
        track: Raw track object with instance_id and candidates
        session_id: Session ID for grouping detections
        first_frame_index: Index of first detection in track
        angle: Card angle (e.g., 'front', 'back')
        frame_entries: List of frame-level detections with normalized images
        canonical_entries: List of canonical (highest-quality) detections
        candidate_hashes: Visual hashes of all candidates for deduplication
        primary_hash: Primary hash of best canonical entry
        side_score: Textiness score for side detection
        appearance_vector: OSNet embedding vector for ReID
        canonical_detection_ids: Set of detection IDs in canonical_entries
        best_canonical_detection_id: Detection ID of highest-scored canonical entry
        fused_canonical: Foil-aware fused image (median or glare-rejection),
                         or fallback to best_canonical normalized if fusion failed.
                         Guaranteed non-None after line 719 fallback check.
        embedding: OSNet ReID embedding of best candidate
        duplicate_track_index: If set, track is duplicate of another in same batch
    """
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
    best_canonical_detection_id: int
    fused_canonical: Optional[np.ndarray]
    embedding: Optional[np.ndarray] = None
    duplicate_track_index: Optional[int] = None

    @property
    def mean_quality_score(self) -> float:
        """Compute mean quality_score across all candidates.

        Extracts quality_score from each candidate's score.components dict.
        Falls back to 0.0 if no candidates have quality_score.
        """
        if not self.track.candidates:
            return 0.0

        quality_scores = []
        for candidate in self.track.candidates:
            # Try to get quality_score from components dict
            if hasattr(candidate, 'score') and hasattr(candidate.score, 'components'):
                quality = candidate.score.components.get('quality_score')
                if quality is not None:
                    quality_scores.append(float(quality))

        if not quality_scores:
            return 0.0
        return float(np.mean(quality_scores))


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
    tracker_backend: str = "botsort"
    centroid_jump_ratio: float = 0.30
    centroid_jump_frames: int = 3
    foil_threshold: float = 50.0
    enable_foil_aware_fusion: bool = True




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
            self.frame_count = self.frames # FORCE ACTIVE

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
            return False # Return False during warmup so we don't accidentally trigger resets
        
        if self._bg_model_u8 is None:
             self._bg_model_u8 = self.background_model.astype(np.uint8)

        diff = cv2.absdiff(gray, self._bg_model_u8)
        return float(np.mean(diff)) < self.threshold


class SessionManager:
    def __init__(self):
        self.active_session_id: Optional[str] = None

    def start_session(self, timestamp: int):
        self.active_session_id = str(timestamp)


class PipelineContext:
    """Per-video context for tracking observed distributions and computing adaptive thresholds.

    Collects:
    - observed_novelty_scores: novelty scores from all processed candidates
    - observed_intra_track_distances: Hamming distances within tracks

    Provides:
    - get_adaptive_novelty_threshold(): p50 clipped ±20% of global
    - get_adaptive_hamming_threshold(): p75*1.2 clipped ±10% of global
    - Graceful fallback to global if <10 samples
    """

    def __init__(self):
        self.observed_novelty_scores: List[float] = []
        self.observed_intra_track_distances: List[float] = []
        self.computer = AdaptiveThresholdComputer()

    def add_novelty_score(self, score: float) -> None:
        """Record a novelty score observed during processing."""
        self.observed_novelty_scores.append(float(score))

    def add_intra_track_distance(self, distance: float) -> None:
        """Record a Hamming distance within a track."""
        self.observed_intra_track_distances.append(float(distance))

    def get_adaptive_novelty_threshold(self, global_threshold: float = 0.08) -> float:
        """Compute adaptive novelty threshold from observed distribution.

        Returns:
            Adaptive threshold (p50 clipped ±20%) or global if insufficient samples.
        """
        return self.computer.compute_novelty_threshold(
            self.observed_novelty_scores,
            global_threshold
        )

    def get_adaptive_hamming_threshold(self, global_threshold: float = 22) -> float:
        """Compute adaptive Hamming threshold from observed distribution.

        Returns:
            Adaptive threshold (p75*1.2 clipped ±10%) or global if insufficient samples.
        """
        return self.computer.compute_hamming_threshold(
            self.observed_intra_track_distances,
            global_threshold
        )


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
        self.tracker = ByteTrackAdapter(min_track_length=12)

    def flush_tracker(self):
        self.tracker.finalize()
        self.tracker.reset()

    def process(self, video_path: Path, options: ProcessingOptions, debug_config: Any = None) -> ProcessingResult:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video does not exist: {video_path}")

        # Initialize per-video context for adaptive thresholds
        pipeline_context = PipelineContext()

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
        t_start = time.time()
        stats, detection_rows = _run_pipeline_workers(
            video_path=video_path,
            video_id=video_id,
            frame_dir=frame_dir,
            sampler=self.sampler,
            detector=self.detector,
            options=options,
        )
        t_ml = time.time() - t_start
        print(f"[Stage: ML Inference] | {t_ml:.2f}s | Detections: {len(detection_rows)}")

        inter_window_gaps = stats.sampler_telemetry.get("last_inter_window_gaps_frames") or []
        video_fps = stats.sampler_telemetry.get("last_source_fps", 30.0)
        valley_split_frames: set[int] = set(stats.sampler_telemetry.get("last_valley_splits") or [])
        gap_dist = compute_session_gap_frames(
            inter_window_gaps,
            fps=video_fps,
        )
        # Use the adaptive gap directly — null_patience_frames is too small (6 frames)
        # to serve as a session gap cap and incorrectly triggers resets on every inter-window
        # gap.  The compute_session_gap_frames already floors at 0.5 s and caps at 3 s.
        effective_session_gap_frames = gap_dist.recommended_gap_frames

        candidates = _build_candidates(detection_rows)

        # v4.1 per-quad background novelty scoring.
        # Build a workspace baseline from the sampler's true empty background proxies.
        # Keep these scores for telemetry/adaptive quality without hard-dropping
        # true cards whose novelty is low against this setup's background.
        bg_paths = stats.sampler_telemetry.get("background_proxies_paths", [])
        _bg_model = None
        if bg_paths:
            _bg_model = BackgroundModel.from_source_frame_paths(bg_paths, n=30)
            
        if _bg_model is not None:
            _scored = _collect_candidate_novelty_scores(
                candidates, _bg_model, context=pipeline_context
            )
            print(
                f"[Stage: NoveltyGate] | scored {_scored} candidates "
                "(hard novelty pruning disabled)"
            )
        else:
            print("[Stage: NoveltyGate] | skipped (no background proxies found)")
        self._bg_model = _bg_model  # retained for downstream scoring/diagnostics
        
        candidate_confidences = [candidate.score.total for candidate in candidates]
        if candidate_confidences:
            # Keep BoT-SORT recall-oriented. The detector's usable card
            # confidences can sit in the 0.60-0.72 range, so allowing the
            # adaptive percentile to rise near 0.75 suppresses real cards.
            tracker_t_high = float(
                np.clip(np.percentile(candidate_confidences, 65), 0.40, 0.60)
            )
        else:
            tracker_t_high = 0.60
        tracker_t_low = max(0.20, tracker_t_high - 0.20)

        # Task 6: Compute inter-detection gaps for adaptive min_track_length
        inter_detection_gaps: list[float] = []
        if len(detection_rows) > 1:
            # Sort detections by frame_index to compute gaps in chronological order
            sorted_detections = sorted(
                detection_rows,
                key=lambda row: int(row.detection_packet.frame_index)
            )
            for i in range(1, len(sorted_detections)):
                prev_frame = int(sorted_detections[i - 1].detection_packet.frame_index)
                curr_frame = int(sorted_detections[i].detection_packet.frame_index)
                gap = curr_frame - prev_frame
                if gap > 0:  # Only positive gaps
                    inter_detection_gaps.append(gap)

        # Task 6: Use adaptive formula instead of len(detection_rows) // 3
        min_track_length_adaptive = adaptive_min_track_length(len(detection_rows), inter_detection_gaps)
        min_track_length_value = min(
            options.min_track_length,
            min_track_length_adaptive
        )

        # lost_track_buffer: keep a ByteTrack "lost" track alive for half the session gap
        # so detections that briefly disappear within a session are re-associated rather
        # than spawning a new track.
        lost_track_buffer = max(options.null_patience_frames * 2, effective_session_gap_frames // 2)

        from .tracking import ByteTrackAdapter, BoTSORTAdapter, CentroidJumpDetector

        if options.tracker_backend == "botsort":
            try:
                self.tracker = BoTSORTAdapter(
                    min_track_length=min_track_length_value,
                    track_activation_threshold=tracker_t_high,
                    lost_track_buffer=lost_track_buffer,
                    minimum_matching_threshold=tracker_t_low,
                )
            except ImportError:
                # boxmot not installed — fall back to ByteTrack with a warning
                import warnings
                warnings.warn(
                    "boxmot not installed, falling back to ByteTrack. "
                    "Install with: pip install 'card-capture[pipeline_v21]'",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.tracker = ByteTrackAdapter(
                    min_track_length=min_track_length_value,
                    lost_track_buffer=lost_track_buffer,
                )
        else:
            self.tracker = ByteTrackAdapter(
                min_track_length=min_track_length_value,
                lost_track_buffer=lost_track_buffer,
            )

        centroid_detector = CentroidJumpDetector(
            jump_ratio=options.centroid_jump_ratio,
            jump_within_frames=options.centroid_jump_frames,
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
        t_track_start = time.time()
        tracker_events: list[dict] = []
        _tracked_instance_ids: set[str] = set()
        for frame_index, timestamp_ms, _ in stats.accepted_frame_presence:
            if last_frame_idx != -1:
                gap = frame_index - last_frame_idx
                valley_in_gap = any(last_frame_idx < vs <= frame_index for vs in valley_split_frames)
                if gap > effective_session_gap_frames or valley_in_gap:
                    reason = "sampled_frame_gap" if gap > effective_session_gap_frames else "valley_split"
                    self.storage.add_pipeline_event(
                        video_id=video_id,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        event_type="session_reset",
                        data={"reason": reason, "gap_frames": gap}
                    )
                    print(f"[Stage: Tracking] | Session: {current_session_id} | Action: Session Reset ({reason}, gap={gap}f)")
                    self.session_manager.active_session_id = None
                    self.tracker.reset()
                    centroid_detector.reset()
                    _tracked_instance_ids.clear()
            last_frame_idx = frame_index

            # --- Centroid jump split ---
            primary_obb_corners = None
            frame_candidates_for_centroid = by_frame.get(frame_index, [])
            if frame_candidates_for_centroid:
                best = max(frame_candidates_for_centroid, key=lambda c: c.score.total)
                if best.corners:
                    primary_obb_corners = np.array(best.corners, dtype=np.float32)
            # frame_width: use a reasonable default; exact value from sampler telemetry if available
            frame_width = getattr(options, '_frame_width_hint', 1280)
            if centroid_detector.update(primary_obb_corners, frame_width):
                self.storage.add_pipeline_event(
                    video_id=video_id,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    event_type="session_reset",
                    data={"reason": "centroid_jump"}
                )
                self.tracker.reset()
                centroid_detector.reset()
                _tracked_instance_ids.clear()
                self.session_manager.active_session_id = None
                current_session_id += 1

            # --- ReID split (BoT-SORT pending_splits) ---
            if hasattr(self.tracker, "pending_splits") and frame_index in self.tracker.pending_splits:
                self.tracker.pending_splits = [
                    fi for fi in self.tracker.pending_splits if fi != frame_index
                ]
                self.storage.add_pipeline_event(
                    video_id=video_id,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    event_type="session_reset",
                    data={"reason": "reid_shift"}
                )
                self.tracker.reset()
                centroid_detector.reset()
                _tracked_instance_ids.clear()
                self.session_manager.active_session_id = None
                current_session_id += 1

            frame_candidates = by_frame.get(frame_index, [])

            if self.session_manager.active_session_id is None:
                self.session_manager.active_session_id = str(timestamp_ms)
                current_session_id += 1
            frame_to_session[frame_index] = current_session_id
            for adapted in self.tracker.process(frame_candidates):
                action = "new_track" if adapted.instance_id not in _tracked_instance_ids else "assigned_existing"
                _tracked_instance_ids.add(adapted.instance_id)
                tracker_events.append({
                    "action": action,
                    "frame_index": int(frame_index),
                    "track_id": adapted.track_id,
                    "instance_id": adapted.instance_id,
                })

        tracks = self.tracker.finalize()
        raw_track_lengths = [len(track.candidates) for track in tracks]
        t_track = time.time() - t_track_start
        print(f"[Stage: Tracking] | {t_track:.2f}s | Tracks Finalized: {len(tracks)}")

        # Stage 4: Refinement (Lazy High-Res Processing)
        t_refine_start = time.time()
        
        # Collect all frame indices needed for canonical views across all tracks
        canonical_indices_to_track: dict[int, list[tuple[Any, list[ScoredCandidate]]]] = {}
        for track in tracks:
            scored_candidates = sorted(track.candidates, key=lambda c: c.score.total, reverse=True)[:8]
            for candidate in scored_candidates:
                canonical_indices_to_track.setdefault(candidate.frame_index, []).append((track, scored_candidates))

        # Sequentially decode only the required high-res frames
        all_required_indices = sorted(canonical_indices_to_track.keys())
        print(f"[Stage: Refinement] | Decoding {len(all_required_indices)} high-res frames sequentially...")

        # We'll use a local dictionary to store the decoded images
        decoded_images: dict[int, np.ndarray] = {}
        if all_required_indices:
            # Use the sampler's decoding logic (which we optimized for sequential reading)
            # We need to reach into the sampler's implementation or replicate the sequential read.
            capture = _open_capture(video_path)
            try:
                curr_idx = 0
                target_idx_set = set(all_required_indices)
                max_target = max(target_idx_set)
                while curr_idx <= max_target:
                    ok, frame = capture.read()
                    if not ok: break
                    if curr_idx in target_idx_set:
                        decoded_images[curr_idx] = frame
                    # Task 7: Refresh background model during detected empty windows
                    # (valley_split_frames from sampler). This tracks slow lighting drift
                    # during long captures using EWMA with alpha=0.1.
                    if self._bg_model is not None and curr_idx in valley_split_frames:
                        self._bg_model.refresh_from_frame(frame)
                    curr_idx += 1
            finally:
                capture.release()

        # Now perform batch warping on the decoded high-res frames
        normalizer = PrecisionNormalizer()
        deduplicator = VisualDeduplicator()
        prepared_tracks: list[_PreparedTrack] = []
        saved_count = 0

        for track in tracks:
            scored_track = sorted(track.candidates, key=lambda c: c.score.total, reverse=True)[:8]
            frame_entries = []
            normalized_by_detection: dict[int, np.ndarray] = {}
            
            if self.kornia_normalizer is not None:
                batch_items: list[tuple[np.ndarray, list[tuple[float, float]]]] = []
                batch_ids: list[int] = []
                for candidate in scored_track:
                    raw_image = decoded_images.get(candidate.frame_index)
                    if raw_image is None:
                        _r = detection_rows[candidate.detection_id]
                        raw_image = np.zeros(
                            (_r.detection_packet.height, _r.detection_packet.width, 3), dtype=np.uint8
                        )
                    batch_items.append((raw_image, candidate.corners))
                    batch_ids.append(candidate.detection_id)
                
                if batch_items:
                    try:
                        warped = self.kornia_normalizer.warp_canonical_batch(batch_items, rotate_180=options.rotate_180)
                        for detection_id, image in zip(batch_ids, warped):
                            normalized_by_detection[detection_id] = image
                    except Exception as e:
                        print(f"Kornia warp failed: {e}")
                        normalized_by_detection = {}

            for candidate in scored_track:
                raw_image = decoded_images.get(candidate.frame_index)
                row = detection_rows[candidate.detection_id]
                if raw_image is None:
                    raw_image = np.zeros(
                        (row.detection_packet.height, row.detection_packet.width, 3), dtype=np.uint8
                    )

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
                    normalized = normalizer.normalize(raw_image, candidate.corners, rotate_180=options.rotate_180)

                # Compute novelty score (how much the candidate differs from background)
                novelty_score = 1.0
                if self._bg_model is not None and candidate.corners:
                    from .presence.background_novelty import quad_novelty
                    novelty_score = quad_novelty(
                        raw_image, candidate.corners, self._bg_model,
                        color_space="lab", lab_weights=(1.0, 0.5, 0.5)
                    )

                # Perform full quality scoring on the rectified crop
                quality_score = self.scorer.score(
                    normalized,
                    float(row.detection_packet.corner_detection.confidence),
                    novelty=novelty_score
                )
                glare_centroid = find_glare_centroid(normalized)
                glare_x, glare_y = glare_centroid if glare_centroid else (None, None)
                frame_entries.append(
                    {
                        "candidate": candidate,
                        "row": row,
                        "normalized": normalized,
                        "quality_score": quality_score,
                        "visual_hash": deduplicator.compute_phash(normalized),
                        "glare_x": glare_x,
                        "glare_y": glare_y,
                        "sharpness": quality_score.components["sharpness"],
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
            best_canonical = max(canonical_entries, key=lambda e: e["quality_score"].total)
            phash = best_canonical["visual_hash"]

            # Stage 9: Foil-aware fusion of canonical frames
            # Extract normalized frames from canonical entries for fusion
            canonical_frames = [entry["normalized"] for entry in canonical_entries]
            fuser = MultiFrameFuser()
            # Use foil_threshold from options; if enable_foil_aware_fusion is False, disable foil detection
            foil_threshold = options.foil_threshold if options.enable_foil_aware_fusion else None
            fused_canonical = fuser.fuse(canonical_frames, foil_threshold=foil_threshold)
            # Ensure fused_canonical is not None (best_canonical should always exist with frames)
            if fused_canonical is None:
                fused_canonical = best_canonical["normalized"]

            candidate_hashes: list[str] = []
            for entry in canonical_entries:
                h = str(entry["visual_hash"])
                candidate_hashes.append(h)
                # Apply 180-rotation check for hashing as well
                rotated = cv2.rotate(entry["normalized"], cv2.ROTATE_180)
                candidate_hashes.append(deduplicator.compute_phash(rotated))

            # Validate consistency: best_canonical_detection_id must be in canonical_detection_ids
            assert best_canonical["candidate"].detection_id in canonical_detection_ids, \
                f"best_canonical detection_id {best_canonical['candidate'].detection_id} not in canonical_detection_ids {canonical_detection_ids}"

            # Extract OSNet embedding from best canonical candidate (Wave 1 ReID)
            embedding = None
            best_candidate = best_canonical["candidate"]
            if hasattr(best_candidate, "reid_embedding") and best_candidate.reid_embedding is not None:
                embedding = best_candidate.reid_embedding
            elif hasattr(track, "reid_embedding") and track.reid_embedding is not None:
                embedding = track.reid_embedding

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
                    best_canonical_detection_id=best_canonical["candidate"].detection_id,
                    fused_canonical=fused_canonical,
                    embedding=embedding,
                )
            )

        t_refine = time.time() - t_refine_start
        print(f"[Stage: Refinement] | {t_refine:.2f}s | Sessions: {current_session_id}")

        if getattr(self, "_bg_model", None) is not None:
            prepared_tracks = _prune_empty_workspace_tracks(
                prepared_tracks, self._bg_model, threshold=0.0, context=pipeline_context
            )

        _resolve_session_tracks(prepared_tracks, deduplicator, context=pipeline_context)

        # Capture hard cases for active learning (Wave 3 Task 3)
        _capture_hard_cases_from_sessions(prepared_tracks, video_id, output_dir)

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
        sampler_telemetry["adaptive_min_track_length"] = min_track_length_value
        sampler_telemetry["detections"] = len(detection_rows)
        sampler_telemetry["raw_track_lengths"] = raw_track_lengths
        sampler_telemetry["tracks_finalized"] = len(tracks)
        sampler_telemetry["track_lengths"] = track_lengths
        sampler_telemetry["duplicate_tracks"] = duplicate_track_count
        sampler_telemetry["tracker_event_count"] = len(tracker_events)
        sampler_telemetry["tracker_event_actions"] = _count_event_values(tracker_events, "action")
        sampler_telemetry["tracker_split_reasons"] = _count_event_values(tracker_events, "split_reason")
        sampler_telemetry["adaptive_gap_p50"] = gap_dist.p50_frames
        sampler_telemetry["adaptive_gap_p95"] = gap_dist.p95_frames
        sampler_telemetry["adaptive_gap_recommended"] = gap_dist.recommended_gap_frames
        sampler_telemetry["adaptive_gap_effective"] = effective_session_gap_frames

        t_storage_start = time.time()
        track_instance_ids: dict[int, int] = {}
        # Keep track of canonical embeddings for the current run to merge cross-session duplicates
        run_canonical_embeddings: list[tuple[int, np.ndarray, str]] = []  # (instance_id, embedding, phash)

        for track_index, prepared in enumerate(prepared_tracks):
            timer = PipelineTimer()
            instance_id = self.storage.add_card_instance(
                video_id=video_id,
                track_id=prepared.track.instance_id,
                angle=prepared.angle,
                session_id=str(prepared.session_id),
            )
            track_instance_ids[track_index] = instance_id

            duplicate_of = None
            if prepared.duplicate_track_index is not None:
                duplicate_of = track_instance_ids.get(prepared.duplicate_track_index)

            # --- Global Deduplication (Cross-Session/Cross-Video) ---
            is_intra_run_match = False
            if duplicate_of is None:
                # 1. Check against current run's canonicals using embeddings (most robust)
                if prepared.embedding is not None:
                    from .identity.embedding_distance import embedding_same_card_score
                    for other_id, other_emb, other_phash in run_canonical_embeddings:
                        if embedding_same_card_score(prepared.embedding, other_emb, threshold=0.5):
                            if deduplicator.hamming_distance(prepared.primary_hash, other_phash) <= 30:
                                duplicate_of = other_id
                                is_intra_run_match = True
                                break
                
                # 2. Fall back to Storage-based lookup (pHash-only, cross-video)
                if duplicate_of is None:
                    duplicate_of = self.storage.find_canonical_for_hashes(
                        prepared.candidate_hashes,
                        threshold=6,
                    )
                    # Note: We don't easily know if this is same video without querying DB,
                    # but typically find_canonical_for_hashes will find the earliest one.
            
            self.storage.update_instance_deduplication(instance_id, prepared.primary_hash, duplicate_of)
            
            # If this is NOT a duplicate, add it to our run's canonical cache
            if duplicate_of is None and prepared.embedding is not None:
                run_canonical_embeddings.append((instance_id, prepared.embedding, prepared.primary_hash))

            rectified_paths: dict[int, str] = {}
            for canonical_order, entry in enumerate(prepared.canonical_entries):
                candidate = entry["candidate"]
                rectified_path = (
                    crops_dir / f"instance_{instance_id}_view_{canonical_order}_rectified.jpg"
                ).resolve()
                # Use fused canonical for best_canonical entry, raw normalized for others
                if candidate.detection_id == prepared.best_canonical_detection_id:
                    assert prepared.fused_canonical is not None, \
                        "fused_canonical must be set for best_canonical_detection_id"
                    cv2.imwrite(str(rectified_path), prepared.fused_canonical)
                else:
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
                    quality_score=entry["quality_score"].components,
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
                raw_image = decoded_images.get(row.detection_packet.frame_index)
                if raw_image is None:
                    raw_image = np.zeros((10, 10, 3), dtype=np.uint8)

                self.storage.add_evidence_frame(
                    card_view_id=view_id,
                    source_frame_path=_persist_source_frame(
                        frame_dir=frame_dir,
                        video_id=video_id,
                        row=row,
                        image=raw_image,
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
                
                # Only suppress saving for intra-run duplicates (secondary sides/Backs 
                # or cross-session matches within the same video).
                # We still want to record 'saved_cards' for global duplicates (same card, 
                # different video) so that per-run recall is accurate.
                
                is_intra_run_duplicate = prepared.duplicate_track_index is not None or is_intra_run_match
                
                if not is_intra_run_duplicate:
                    self.storage.add_saved_card(
                        detection_id=best_view_id,
                        image_path=canonical_image_path,
                        final_score=float(prepared.canonical_entries[0]["candidate"].score.total),
                    )
                    saved_count += 1

        t_storage = time.time() - t_storage_start

        print(f"\n--- Performance Summary ---")
        print(f"ML Inference: {t_ml:.2f}s")
        print(f"Tracking:     {t_track:.2f}s")
        print(f"Refinement:   {t_refine:.2f}s")
        print(f"Storage/Dedup: {t_storage:.2f}s")
        print(f"Total:        {time.time() - t_start:.2f}s")
        print(f"---------------------------\n")

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
        
        # Persist events to DB for UI
        for event in tracker_events:
            if event["action"] in ("new_track", "reset"):
                self.storage.add_pipeline_event(
                    video_id=video_id,
                    frame_index=event.get("frame_index", 0),
                    timestamp_ms=event.get("timestamp_ms", 0),
                    event_type=f"tracker_{event['action']}",
                    data={"reason": event.get("split_reason"), "track_id": event.get("assigned_track_id")}
                )

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
    image: np.ndarray,
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
    cv2.imwrite(str(path), image)
    persisted = str(path)
    persisted_frames[key] = persisted
    return persisted





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


def _track_representative_hash(
    track: Any,
    deduplicator: VisualDeduplicator,
) -> Optional[str]:
    """Compute a phash from the highest-confidence candidate in the track."""
    best = None
    best_score = -1.0
    for cand in track.candidates:
        if cand.image_path is None:
            continue
        score = float(cand.score.total)
        if score > best_score:
            best = cand
            best_score = score
    if best is None:
        return None
    img = cv2.imread(best.image_path)
    if img is None:
        return None
    try:
        return deduplicator.compute_phash(img)
    except Exception:
        return None


def _compute_quality_weighted_score(prepared: _PreparedTrack, max_length: int) -> float:
    """Compute composite track selection score.

    Args:
        prepared: _PreparedTrack to score
        max_length: Maximum track length in session (for normalization)

    Returns:
        Composite score: 0.3 * normalized_length + 0.7 * mean_quality_score
        (Quality is weighted higher to prefer sharp short tracks over blurry long)
    """
    if max_length <= 0:
        norm_length = 0.0
    else:
        norm_length = len(prepared.track.candidates) / max_length

    mean_quality = prepared.mean_quality_score
    return 0.3 * norm_length + 0.7 * mean_quality


def _is_reid_duplicate(
    phash_1: str,
    phash_2: str,
    embedding_1: Optional[np.ndarray],
    embedding_2: Optional[np.ndarray],
    deduplicator: VisualDeduplicator,
) -> bool:
    """Determine if two canonicals represent the same card across videos (cross-video dedup).

    Uses a two-stage approach:
    1. **pHash pre-filter (broad candidates, fast):** Hamming distance > 22 → definitely not same card.
    2. **Embeddings as deciding metric (discriminative, view-invariant):**
       - If embeddings available: use cosine distance < 0.5 for same-card classification.
       - If embeddings unavailable: use stricter pHash threshold (≤15) for backward compat.

    Args:
        phash_1: pHash of first canonical (hex string).
        phash_2: pHash of second canonical (hex string).
        embedding_1: OSNet embedding of first canonical (shape [256,]) or None.
        embedding_2: OSNet embedding of second canonical (shape [256,]) or None.
        deduplicator: VisualDeduplicator instance for pHash distance computation.

    Returns:
        True if same card (duplicate), False otherwise.

    Wave 3: Cross-video dedup via embedding distance, replacing pHash-only logic.
    Backward compatible: falls back to stricter pHash threshold if embeddings unavailable.
    """
    from .identity.embedding_distance import embedding_same_card_score

    # Stage 1: pHash as broad pre-filter (Wave 3)
    ham = deduplicator.hamming_distance(phash_1, phash_2)
    
    # If embeddings available: use them as the primary decision metric.
    # Relax pHash threshold when embeddings are present to handle blur/deformation.
    if embedding_1 is not None and embedding_2 is not None:
        if ham > 28: # Much more relaxed than 22
            return False
        from .identity.embedding_distance import embedding_same_card_score
        return embedding_same_card_score(embedding_1, embedding_2, threshold=0.5)
    else:
        # Embeddings unavailable: fall back to stricter pHash threshold for backward compat
        return ham <= 15


def _resolve_session_tracks(
    prepared_tracks: list[_PreparedTrack],
    deduplicator: VisualDeduplicator,
    context: Optional[PipelineContext] = None,
) -> None:
    """For each session, label tracks using side_score (textiness) as primary sort.

    High textiness (≥0.5) tracks are Front (image-rich sides). Low textiness
    tracks may be promoted to Back if they represent the same physical card.
    Uses OSNet cosine embeddings for same-card detection (Wave 3), with fallback
    to pHash Hamming distance if embeddings unavailable (backward compat).

    If textiness margin is within _SESSION_TEXTINESS_MARGIN, the lower-textiness
    track is admitted as Back even if it would normally be its own Front.

    Quality-weighted tie-breaker: When side_score is similar, composite score
    (0.6·normalized_length + 0.4·mean_quality_score) is used to prefer sharp
    short tracks over blurry long tracks.

    If context is provided, records observed Hamming distances within tracks for
    adaptive threshold computation.
    """
    from .identity.embedding_distance import embedding_same_card_score

    by_session: dict[int, list[tuple[int, _PreparedTrack]]] = {}
    for track_index, prepared in enumerate(prepared_tracks):
        by_session.setdefault(prepared.session_id, []).append((track_index, prepared))

    for session_tracks in by_session.values():
        if not session_tracks:
            continue

        # Find max length in session for normalized_length computation
        max_length = max(len(item[1].track.candidates) for item in session_tracks) if session_tracks else 1

        # Primary sort: by side_score (descending) - high textiness first.
        # Secondary tie-breaker: quality-weighted composite score (descending)
        session_tracks.sort(
            key=lambda item: (
                -item[1].side_score,
                -_compute_quality_weighted_score(item[1], max_length)
            )
        )

        first_index, first_prepared = session_tracks[0]
        first_prepared.duplicate_track_index = None
        first_prepared.angle = "Front"

        # FIRST PASS: Collect intra-track Hamming distances for adaptive threshold
        # This allows later comparisons to use a threshold adapted from actual distribution
        if context is not None:
            for other_index, other_prepared in session_tracks[1:]:
                # Collect pHash distances (both embedding and pHash paths)
                first_hash = _track_representative_hash(first_prepared.track, deduplicator)
                other_hash = _track_representative_hash(other_prepared.track, deduplicator)
                if first_hash is not None and other_hash is not None:
                    hamming_distance = deduplicator.hamming_distance(first_hash, other_hash)
                    context.add_intra_track_distance(float(hamming_distance))

        # SECOND PASS: Apply same-card classification using adaptive threshold
        for other_index, other_prepared in session_tracks[1:]:
            same_card = False
            hamming_distance = None

            # Try embedding-based same-card detection (Wave 1/3)
            if (first_prepared.embedding is not None and
                other_prepared.embedding is not None):
                # Use OSNet embeddings with threshold 0.5
                same_card = embedding_same_card_score(
                    first_prepared.embedding,
                    other_prepared.embedding,
                    threshold=0.5
                )
            else:
                # Fallback to pHash if embeddings unavailable
                first_hash = _track_representative_hash(first_prepared.track, deduplicator)
                other_hash = _track_representative_hash(other_prepared.track, deduplicator)
                if first_hash is not None and other_hash is not None:
                    hamming_distance = deduplicator.hamming_distance(first_hash, other_hash)
                    # Use adaptive threshold if context is available, otherwise fall back to global constant
                    threshold = (_SAME_CARD_HAMMING_MAX
                                 if context is None
                                 else context.get_adaptive_hamming_threshold(_SAME_CARD_HAMMING_MAX))
                    same_card = hamming_distance <= threshold

            if same_card:
                # Check textiness margin: if within threshold, admit as Back
                textiness_margin = first_prepared.side_score - other_prepared.side_score
                if textiness_margin <= _SESSION_TEXTINESS_MARGIN:
                    # Textiness is similar enough; promote to Back if it's the first Back.
                    if all(prev_pt.duplicate_track_index != first_index or prev_pt.angle != "Back"
                           for _, prev_pt in session_tracks[1:]
                           if prev_pt is not other_prepared):
                        other_prepared.angle = "Back"
                    else:
                        other_prepared.angle = "Front"
                else:
                    # Textiness margin too large; this is likely a different side of the same card.
                    if all(prev_pt.duplicate_track_index != first_index or prev_pt.angle != "Back"
                           for _, prev_pt in session_tracks[1:]
                           if prev_pt is not other_prepared):
                        other_prepared.angle = "Back"
                    else:
                        other_prepared.angle = "Front"
                other_prepared.duplicate_track_index = first_index
            else:
                # Distinct card → independent Front instance.
                other_prepared.angle = "Front"
                other_prepared.duplicate_track_index = None


def _capture_hard_cases_from_sessions(
    prepared_tracks: list[_PreparedTrack],
    video_id: int,
    output_dir: Path,
) -> None:
    """Capture hard cases for active learning from session tracks.

    Groups tracks by session and checks each session for hard-case criteria:
    - Multiple Fronts (>2)
    - Borderline Hamming distance ([18, 26])
    - Borderline confidence ([0.45, 0.55])
    - Low border purity (<0.3)

    Args:
        prepared_tracks: List of prepared tracks after session resolution
        video_id: Video ID for metadata
        output_dir: Output directory for hard_cases.jsonl
    """
    # Group tracks by session
    by_session: dict[int, list[_PreparedTrack]] = {}
    for prepared in prepared_tracks:
        by_session.setdefault(prepared.session_id, []).append(prepared)

    hard_cases_file = output_dir / "hard_cases.jsonl"

    # Process each session
    for session_id, session_tracks in by_session.items():
        if not session_tracks:
            continue

        # Build session dict for is_hard_case
        front_tracks = []
        for prepared in session_tracks:
            front_tracks.append({"track_id": str(prepared.track.instance_id), "angle": prepared.angle})

        # Extract hamming distances from all candidates in the session
        hamming_values = []
        confidence_scores = []
        border_purity_scores = []

        for prepared in session_tracks:
            # Compute Hamming distances between this track and others in session
            for other in session_tracks:
                if prepared is other:
                    continue
                # Use representative hashes
                hash_a = prepared.primary_hash
                hash_b = other.primary_hash
                if hash_a and hash_b:
                    dedup = VisualDeduplicator()
                    ham = dedup.hamming_distance(hash_a, hash_b)
                    hamming_values.append(ham)

            # Extract confidence scores from candidates
            for candidate in prepared.track.candidates:
                if hasattr(candidate, 'score') and hasattr(candidate.score, 'total'):
                    confidence_scores.append(float(candidate.score.total))

            # Extract border purity from candidates
            for candidate in prepared.track.candidates:
                if hasattr(candidate, 'score') and hasattr(candidate.score, 'components'):
                    border_purity = candidate.score.components.get('border_purity')
                    if border_purity is not None:
                        border_purity_scores.append(float(border_purity))

        session = {
            "video_id": video_id,
            "session_id": str(session_id),
            "front_tracks": front_tracks,
            "hamming_values": hamming_values,
            "confidence_scores": confidence_scores,
            "border_purity_scores": border_purity_scores,
        }

        # Check if hard case
        reason = is_hard_case(session)
        if reason:
            capture_hard_case(session, reason, output_file=str(hard_cases_file))

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


def _filter_candidates_by_novelty(
    candidates: list[ScoredCandidate],
    bg: BackgroundModel,
    threshold: float = 0.08,
    context: Optional[PipelineContext] = None,
) -> list[ScoredCandidate]:
    """Drop candidates whose quad interior matches the workspace baseline.

    Loads each unique source frame at most once (small per-frame cost: a single
    grayscale conversion via cv2.imread).

    If context is provided, records observed novelty scores for adaptive threshold
    computation and uses adaptive threshold if sufficient samples available.
    """
    from .presence.background_novelty import quad_novelty

    frame_cache: dict[str, np.ndarray] = {}
    kept: list[ScoredCandidate] = []

    # Determine which threshold to use (adaptive or global)
    effective_threshold = threshold
    if context is not None:
        # First pass: collect all scores to determine adaptive threshold
        collected_scores: list[float] = []
        for cand in candidates:
            if not cand.corners or cand.image_path is None:
                continue
            img = frame_cache.get(cand.image_path)
            if img is None:
                img = cv2.imread(cand.image_path)
                if img is None:
                    continue
                frame_cache[cand.image_path] = img
            novelty = quad_novelty(img, cand.corners, bg, color_space="lab", lab_weights=(1.0, 0.5, 0.5))
            collected_scores.append(novelty)

        # SOLE COLLECTION POINT: Only candidate-level gate collects novelty scores.
        # Track-level pruning reads the computed threshold but does NOT re-collect scores.
        # This ensures len(context.observed_novelty_scores) == N (not 2N).
        if context is not None:
            for novelty in collected_scores:
                context.add_novelty_score(novelty)

        # Use adaptive threshold if we have enough data
        if len(collected_scores) >= 10:
            effective_threshold = context.get_adaptive_novelty_threshold(threshold)

    # Second pass: filter using effective threshold
    frame_cache.clear()  # Clear and refill for consistency
    for cand in candidates:
        if not cand.corners or cand.image_path is None:
            kept.append(cand)
            continue
        img = frame_cache.get(cand.image_path)
        if img is None:
            img = cv2.imread(cand.image_path)
            if img is None:
                kept.append(cand)  # cannot evaluate; let downstream stages decide
                continue
            frame_cache[cand.image_path] = img
        novelty = quad_novelty(img, cand.corners, bg, color_space="lab", lab_weights=(1.0, 0.5, 0.5))
        if novelty >= effective_threshold:
            kept.append(cand)
    return kept


def _collect_candidate_novelty_scores(
    candidates: list[ScoredCandidate],
    bg: BackgroundModel,
    context: Optional[PipelineContext] = None,
) -> int:
    """Collect candidate novelty scores without dropping pre-tracking candidates."""
    if context is None:
        return 0

    from .presence.background_novelty import quad_novelty

    frame_cache: dict[str, np.ndarray] = {}
    scored_count = 0
    for cand in candidates:
        if not cand.corners or cand.image_path is None:
            continue
        img = frame_cache.get(cand.image_path)
        if img is None:
            img = cv2.imread(cand.image_path)
            if img is None:
                continue
            frame_cache[cand.image_path] = img
        novelty = quad_novelty(img, cand.corners, bg, color_space="lab", lab_weights=(1.0, 0.5, 0.5))
        context.add_novelty_score(novelty)
        scored_count += 1
    return scored_count


def _prune_empty_workspace_tracks(
    prepared_tracks: list[_PreparedTrack],
    bg: BackgroundModel,
    threshold: float = 0.08,
    context: Optional[PipelineContext] = None,
) -> list[_PreparedTrack]:
    """Drop tracks whose median-novelty across candidates falls below threshold.

    A real card's quad is always novel; an empty-stand track's quad is, by
    definition, the workspace itself. We use the median so a few sharp
    outliers (e.g., a card briefly entering frame) cannot rescue a track
    that is overwhelmingly the empty workspace.

    If context is provided, records observed novelty scores and uses adaptive
    threshold if sufficient samples available.
    """
    from .presence.background_novelty import quad_novelty

    frame_cache: dict[str, np.ndarray] = {}
    kept: list[_PreparedTrack] = []

    # Collect all novelties for adaptive threshold computation
    effective_threshold = threshold
    if context is not None:
        all_novelties: list[float] = []
        for pt in prepared_tracks:
            novelties: list[float] = []
            for cand in pt.track.candidates:
                if not cand.corners or cand.image_path is None:
                    continue
                img = frame_cache.get(cand.image_path)
                if img is None:
                    img = cv2.imread(cand.image_path)
                    if img is None:
                        continue
                    frame_cache[cand.image_path] = img
                novelty = quad_novelty(img, cand.corners, bg, color_space="lab", lab_weights=(1.0, 0.5, 0.5))
                novelties.append(novelty)
                all_novelties.append(novelty)

        # Use adaptive threshold if sufficient samples
        if len(all_novelties) >= 10:
            effective_threshold = context.get_adaptive_novelty_threshold(threshold)
        frame_cache.clear()

    for pt in prepared_tracks:
        novelties: list[float] = []
        for cand in pt.track.candidates:
            if not cand.corners or cand.image_path is None:
                continue
            img = frame_cache.get(cand.image_path)
            if img is None:
                img = cv2.imread(cand.image_path)
                if img is None:
                    continue
                frame_cache[cand.image_path] = img
            novelties.append(quad_novelty(img, cand.corners, bg, color_space="lab", lab_weights=(1.0, 0.5, 0.5)))
        if not novelties:
            kept.append(pt)
            continue
        median = float(np.median(novelties))
        if median >= effective_threshold:
            kept.append(pt)
        else:
            print(f"[Stage: NoveltyPrune] | dropped track {pt.track.instance_id} "
                  f"(median quad novelty={median:.3f} < {effective_threshold})")
    return kept


def _select_canonical_entries(frame_entries: list[dict], deduplicator: VisualDeduplicator) -> list[dict]:
    if not frame_entries:
        return []

    scored = sorted(
        frame_entries,
        key=_entry_quality_total,
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
                -_entry_quality_total(entry),
            ),
        )[: min(_CANONICAL_MAX_FRAMES, len(frame_entries))]

    ranked = sorted(
        same_appearance,
        key=_entry_quality_total,
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
                _entry_quality_total(entry),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_entry = entry
        if best_entry is None:
            break
        selected.append(best_entry)

    return selected


def _entry_quality_total(entry: dict) -> float:
    quality_score = entry.get("quality_score")
    if quality_score is not None:
        return float(quality_score.total)
    candidate = entry.get("candidate")
    if candidate is not None:
        return float(candidate.score.total)
    return 0.0


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

                accepted_frame_presence.append((frame.frame_index, frame.timestamp_ms, False))
                accepted_frame_count += 1
                t_ingest = timer.timings.get("t_ingest", 0.0)
                t_io = 0.0

                # Resize to proxy for detection queue
                h, w = frame.image.shape[:2]
                scale = 640 / w
                proxy_h = int(round(h * scale))
                proxy_image = cv2.resize(frame.image, (640, proxy_h))

                _put_or_fail(
                    frame_queue,
                    _FrameEnvelope(
                        frame_packet=FramePacket(
                            frame_index=frame.frame_index,
                            timestamp_ms=frame.timestamp_ms,
                            image=proxy_image,
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
    output_dir: Path,
    video_id: int,
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
                    output_dir,
                    video_id,
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
                    output_dir,
                    video_id,
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
    output_dir: Path,
    video_id: int,
    detection_queue,
) -> None:
    if not batch:
        return

    # Prepare frames directory
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

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
