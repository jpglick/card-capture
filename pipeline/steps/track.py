"""Step 3 — track: Stage 5 session-aware tracking.

Replays the accepted frame sequence through ByteTrack / BoT-SORT, emitting
session resets on frame-gap, valley-split, centroid-jump, and ReID-shift
signals.  Returns serialisable track data (no numpy arrays).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pipeline.steps.start import RunContext
from pipeline.steps.novelty import NoveltyOutput


@dataclass
class TrackOutput:
    """Outputs of the track step."""

    # Serialisable track records
    tracks_data: List[Dict[str, Any]]
    # frame_index → session_id mapping
    frame_to_session: Dict[str, int]   # keys are str because JSON
    # Events logged to the pipeline_events table
    tracker_events: List[Dict[str, Any]]
    # Same detection rows (unchanged) forwarded for refine step
    detection_rows: List[Dict[str, Any]]
    sampler_telemetry: Dict[str, Any]
    bg_model_path: Optional[str]
    accepted_frame_presence: List[Tuple[int, int, bool]]
    frame_count: int
    accepted_frame_count: int
    video_id: int


def run(ctx: RunContext, novelty_out: NoveltyOutput) -> TrackOutput:
    """Run session-aware tracking over the accepted frame sequence.

    Args:
        ctx:         RunContext from the start step.
        novelty_out: Output from the novelty step.

    Returns:
        ``TrackOutput`` with finalised tracks and session metadata.
    """
    import numpy as np
    from card_capture.pipeline import (
        ProcessingOptions,
        _build_candidates,
        adaptive_min_track_length,
    )
    from card_capture.adaptive_gap import compute_session_gap_frames
    from card_capture.storage import Storage

    detection_rows = novelty_out.detection_rows
    sampler_telemetry = novelty_out.sampler_telemetry
    accepted_frame_presence = novelty_out.accepted_frame_presence
    video_id = novelty_out.video_id

    # Reconstruct _DetectionEnvelope-compatible objects from the serialised dicts
    from card_capture.models import CornerDetection as ModelCornerDetection, DetectionPacket, PerformanceTelemetry
    from card_capture.pipeline import _DetectionEnvelope

    raw_rows = []
    for row in detection_rows:
        cd = ModelCornerDetection(
            corners=row["corners"],
            confidence=row["confidence"],
        )
        dp = DetectionPacket(
            frame_index=row["frame_index"],
            timestamp_ms=row["timestamp_ms"],
            width=row["width"],
            height=row["height"],
            corner_detection=cd,
            telemetry=PerformanceTelemetry(),
        )
        raw_rows.append(
            _DetectionEnvelope(
                detection_packet=dp,
                source_frame_path=row["source_frame_path"],
                triage_metrics=row.get("triage_metrics", {}),
            )
        )

    candidates = _build_candidates(raw_rows)

    # Adaptive tracker thresholds
    candidate_confidences = [c.score.total for c in candidates]
    if candidate_confidences:
        tracker_t_high = float(np.clip(np.percentile(candidate_confidences, 65), 0.40, 0.60))
    else:
        tracker_t_high = 0.60
    tracker_t_low = max(0.20, tracker_t_high - 0.20)

    inter_detection_gaps: List[float] = []
    if len(detection_rows) > 1:
        sorted_rows = sorted(detection_rows, key=lambda r: int(r["frame_index"]))
        for i in range(1, len(sorted_rows)):
            gap = int(sorted_rows[i]["frame_index"]) - int(sorted_rows[i - 1]["frame_index"])
            if gap > 0:
                inter_detection_gaps.append(gap)

    min_track_length_adaptive = adaptive_min_track_length(len(detection_rows), inter_detection_gaps)
    min_track_length_value = min(ctx.min_track_length, min_track_length_adaptive)

    inter_window_gaps = sampler_telemetry.get("last_inter_window_gaps_frames") or []
    video_fps = sampler_telemetry.get("last_source_fps", 30.0)
    valley_split_frames: Set[int] = set(sampler_telemetry.get("last_valley_splits") or [])
    gap_dist = compute_session_gap_frames(inter_window_gaps, fps=video_fps)
    effective_session_gap_frames = gap_dist.recommended_gap_frames

    lost_track_buffer = max(ctx.null_patience_frames * 2, effective_session_gap_frames // 2)

    from card_capture.tracking import ByteTrackAdapter, CentroidJumpDetector
    tracker = _build_tracker(ctx, min_track_length_value, tracker_t_high, tracker_t_low, lost_track_buffer)
    centroid_detector = CentroidJumpDetector(
        jump_ratio=ctx.centroid_jump_ratio,
        jump_within_frames=ctx.centroid_jump_frames,
    )

    # Group candidates by frame
    by_frame: Dict[int, list] = {}
    for candidate in candidates:
        key = -1 if candidate.frame_index is None else int(candidate.frame_index)
        by_frame.setdefault(key, []).append(candidate)

    storage = Storage(Path(ctx.db_path))

    current_session_id = 0
    frame_to_session: Dict[int, int] = {}
    last_frame_idx = -1
    tracker_events: List[Dict[str, Any]] = []
    tracked_instance_ids: Set[str] = set()
    session_manager_active: Optional[str] = None

    t_track_start = time.time()
    for frame_index, timestamp_ms, _ in accepted_frame_presence:
        if last_frame_idx != -1:
            gap = frame_index - last_frame_idx
            valley_in_gap = any(last_frame_idx < vs <= frame_index for vs in valley_split_frames)
            if gap > effective_session_gap_frames or valley_in_gap:
                reason = "sampled_frame_gap" if gap > effective_session_gap_frames else "valley_split"
                storage.add_pipeline_event(
                    video_id=video_id, frame_index=frame_index,
                    timestamp_ms=timestamp_ms, event_type="session_reset",
                    data={"reason": reason, "gap_frames": gap},
                )
                print(f"[Stage: Tracking] | Session: {current_session_id} | Action: Session Reset ({reason})")
                session_manager_active = None
                tracker.reset()
                centroid_detector.reset()
                tracked_instance_ids.clear()
        last_frame_idx = frame_index

        # Centroid jump split
        frame_candidates_for_centroid = by_frame.get(frame_index, [])
        primary_obb_corners = None
        if frame_candidates_for_centroid:
            best = max(frame_candidates_for_centroid, key=lambda c: c.score.total)
            if best.corners:
                primary_obb_corners = np.array(best.corners, dtype=np.float32)
        frame_width = 1280
        if centroid_detector.update(primary_obb_corners, frame_width):
            storage.add_pipeline_event(
                video_id=video_id, frame_index=frame_index,
                timestamp_ms=timestamp_ms, event_type="session_reset",
                data={"reason": "centroid_jump"},
            )
            tracker.reset()
            centroid_detector.reset()
            tracked_instance_ids.clear()
            session_manager_active = None
            current_session_id += 1

        # ReID split
        if hasattr(tracker, "pending_splits") and frame_index in tracker.pending_splits:
            tracker.pending_splits = [fi for fi in tracker.pending_splits if fi != frame_index]
            storage.add_pipeline_event(
                video_id=video_id, frame_index=frame_index,
                timestamp_ms=timestamp_ms, event_type="session_reset",
                data={"reason": "reid_shift"},
            )
            tracker.reset()
            centroid_detector.reset()
            tracked_instance_ids.clear()
            session_manager_active = None
            current_session_id += 1

        frame_candidates = by_frame.get(frame_index, [])
        if session_manager_active is None:
            session_manager_active = str(timestamp_ms)
            current_session_id += 1
        frame_to_session[frame_index] = current_session_id

        for adapted in tracker.process(frame_candidates):
            action = "new_track" if adapted.instance_id not in tracked_instance_ids else "assigned_existing"
            tracked_instance_ids.add(adapted.instance_id)
            tracker_events.append({
                "action": action,
                "frame_index": int(frame_index),
                "track_id": adapted.track_id,
                "instance_id": adapted.instance_id,
            })

    tracks = tracker.finalize()
    print(f"[Stage: Tracking] | {time.time()-t_track_start:.2f}s | Tracks Finalized: {len(tracks)}")

    # Serialise tracks
    tracks_data: List[Dict[str, Any]] = []
    for track in tracks:
        candidate_ids = [c.detection_id for c in track.candidates]
        first_frame = -1 if not track.candidates or track.candidates[0].frame_index is None else int(track.candidates[0].frame_index)
        tracks_data.append({
            "instance_id": track.instance_id,
            "track_id": getattr(track, "track_id", 0),
            "angle": track.angle,
            "candidate_detection_ids": candidate_ids,
            "first_frame_index": first_frame,
            "session_id": frame_to_session.get(first_frame, 0),
            # Store all candidate details for the refine step
            "candidates": [
                {
                    "detection_id": c.detection_id,
                    "frame_index": c.frame_index,
                    "timestamp_ms": c.timestamp_ms,
                    "image_path": c.image_path,
                    "confidence": c.score.total,
                    "score_total": c.score.total,
                    "score_components": dict(c.score.components),
                    "corners": [(float(x), float(y)) for x, y in c.corners] if c.corners else [],
                }
                for c in track.candidates
            ],
            "reid_embedding": (
                track.reid_embedding.tolist()
                if hasattr(track, "reid_embedding") and track.reid_embedding is not None
                else None
            ),
        })

    # Forward sampler_telemetry with adaptive threshold info
    sampler_telemetry = dict(novelty_out.sampler_telemetry)
    sampler_telemetry["tracker_t_high"] = tracker_t_high
    sampler_telemetry["tracker_t_low"] = tracker_t_low
    sampler_telemetry["adaptive_min_track_length"] = min_track_length_value
    sampler_telemetry["adaptive_gap_recommended"] = gap_dist.recommended_gap_frames
    sampler_telemetry["adaptive_gap_effective"] = effective_session_gap_frames

    return TrackOutput(
        tracks_data=tracks_data,
        frame_to_session={str(k): v for k, v in frame_to_session.items()},
        tracker_events=tracker_events,
        detection_rows=detection_rows,
        sampler_telemetry=sampler_telemetry,
        bg_model_path=novelty_out.bg_model_path,
        accepted_frame_presence=accepted_frame_presence,
        frame_count=novelty_out.frame_count,
        accepted_frame_count=novelty_out.accepted_frame_count,
        video_id=video_id,
    )


def _build_tracker(ctx: RunContext, min_track_length: int, t_high: float, t_low: float, lost_track_buffer: int):
    """Build the appropriate tracker from RunContext."""
    from card_capture.tracking import ByteTrackAdapter

    if ctx.tracker_backend == "botsort":
        try:
            from card_capture.tracking import BoTSORTAdapter
            return BoTSORTAdapter(
                min_track_length=min_track_length,
                track_activation_threshold=t_high,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=t_low,
            )
        except ImportError:
            import warnings
            warnings.warn(
                "boxmot not installed, falling back to ByteTrack.",
                RuntimeWarning, stacklevel=2,
            )
    return ByteTrackAdapter(
        min_track_length=min_track_length,
        lost_track_buffer=lost_track_buffer,
    )
