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

# Average-hash Hamming distance (of 64 bits) above which the primary detection's
# appearance is considered to have *left* the currently-tracked card — a candidate
# same-position swap that motion-only tracking and centroid-jump cannot see.
# Tuned for 8x8 aHash. Note: frame-to-frame aHash noise for a *static* card is
# mean ~9 / p90 ~18 bits, so this threshold alone cannot separate swaps from noise
# — see _CARD_SWAP_PERSIST_FRAMES.
_CARD_SWAP_AHASH_BITS = 18

# A real same-position swap settles into a *new, stable* appearance; a one-frame
# glare/hand/ROI-jitter spike reverts immediately. We therefore require the new
# appearance to persist across this many consecutive single-card frames before
# committing a session reset. This is the actual noise rejector: without it, the
# threshold sits at the p90 of normal noise and fires ~171×/video (5x track
# over-fragmentation). With persistence the reference also drifts on consistent
# frames so slow rotation/lighting changes don't trip it.
_CARD_SWAP_PERSIST_FRAMES = 3


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
    from card_capture.workers import ProcessingOptions, _DetectionEnvelope
    from card_capture.pipeline_utils import _build_candidates, adaptive_min_track_length
    from card_capture.adaptive_gap import compute_session_gap_frames
    from card_capture.storage import Storage

    detection_rows = novelty_out.detection_rows
    sampler_telemetry = novelty_out.sampler_telemetry
    accepted_frame_presence = novelty_out.accepted_frame_presence
    video_id = novelty_out.video_id

    # Reconstruct _DetectionEnvelope-compatible objects from the serialised dicts
    from card_capture.models import CornerDetection as ModelCornerDetection, DetectionPacket, PerformanceTelemetry

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
                source_frame_path=row.get("source_frame_path", ""),
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

    # Per-frame primary appearance hash: the highest-confidence detection's
    # bbox aHash (computed during detect). Drives the card-swap reset below.
    from card_capture.frame_quality import ahash_hamming
    frame_primary_ahash: Dict[int, int] = {}
    _best_conf: Dict[int, float] = {}
    for row in detection_rows:
        fi = row.get("frame_index")
        ah = row.get("triage_metrics", {}).get("ahash")
        if fi is None or ah is None:
            continue
        fi = int(fi)
        conf = float(row.get("confidence", 0.0))
        if conf >= _best_conf.get(fi, -1.0):
            _best_conf[fi] = conf
            frame_primary_ahash[fi] = int(ah)

    # Corners are full-res pixel coords, so centroid-jump must scale by the real
    # frame width (was hard-coded to 1280, mis-scaling the 0.30x jump threshold).
    _frame_width_px = int(detection_rows[0]["width"]) if detection_rows else 3840

    storage = Storage(Path(ctx.db_path))

    current_session_id = 0
    frame_to_session: Dict[int, int] = {}
    last_frame_idx = -1
    # Card-swap appearance state: a trailing reference hash for the card currently
    # being tracked, plus a buffer of consecutive frames whose appearance has left
    # that reference (a *candidate* swap awaiting persistence confirmation).
    swap_ref_ahash: Optional[int] = None
    swap_pending: List[int] = []
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

        # Card-swap split (appearance): a new card placed in the same spot keeps
        # the same centroid and produces no frame gap, so neither the gap nor the
        # centroid-jump signal fires. A sharp change in the primary detection's
        # appearance hash is the only tell — split so each card gets its own track.
        frame_candidates = by_frame.get(frame_index, [])
        cur_ahash = frame_primary_ahash.get(frame_index)
        
        # Only perform the global card_swap reset if we are tracking a single clear primary object.
        # If there are multiple candidates, the 'primary' confidence can alternate between them, 
        # causing false-positive card_swaps on every frame.
        if cur_ahash is not None and len(frame_candidates) <= 1:
            if swap_ref_ahash is None:
                swap_ref_ahash = cur_ahash
            elif ahash_hamming(cur_ahash, swap_ref_ahash) < _CARD_SWAP_AHASH_BITS:
                # Appearance still matches the tracked card. Drift the reference so
                # gradual rotation/lighting changes don't accumulate into a swap.
                swap_ref_ahash = cur_ahash
                swap_pending = []
            else:
                # Appearance left the reference — a *candidate* swap. A one-frame
                # glare/jitter spike reverts (the next frame matches the reference
                # again and clears the buffer); a real swap settles into a stable
                # new appearance and stays. Require the new appearance to persist,
                # and to be internally stable, before committing the reset.
                if swap_pending and ahash_hamming(cur_ahash, swap_pending[0]) >= _CARD_SWAP_AHASH_BITS:
                    swap_pending = [cur_ahash]   # not yet a stable new card; restart
                else:
                    swap_pending.append(cur_ahash)
                if len(swap_pending) >= _CARD_SWAP_PERSIST_FRAMES:
                    storage.add_pipeline_event(
                        video_id=video_id, frame_index=frame_index,
                        timestamp_ms=timestamp_ms, event_type="session_reset",
                        data={"reason": "card_swap"},
                    )
                    print(f"[Stage: Tracking] | Session: {current_session_id} | Action: Session Reset (card_swap)")
                    tracker.reset()
                    centroid_detector.reset()
                    tracked_instance_ids.clear()
                    session_manager_active = None
                    swap_ref_ahash = cur_ahash
                    swap_pending = []
        elif cur_ahash is not None and len(frame_candidates) > 1:
            # Multiple candidates: the 'primary' can alternate between them, so we
            # can't trust the appearance signal. Refresh the reference for when it
            # returns to a single card, but never reset on a multi-candidate frame.
            swap_ref_ahash = cur_ahash
            swap_pending = []

        # Centroid jump split
        primary_obb_corners = None
        if frame_candidates:
            best = max(frame_candidates, key=lambda c: c.score.total)
            if best.corners:
                primary_obb_corners = np.array(best.corners, dtype=np.float32)
        frame_width = _frame_width_px
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
        
        # Collect intra-track Hamming distances for adaptive thresholding
        from card_capture.deduplicator import VisualDeduplicator
        dedup = VisualDeduplicator()
        hashes = [c.visual_hash for c in track.candidates if hasattr(c, "visual_hash") and c.visual_hash]
        # In v4, ScoredCandidate might not have visual_hash yet if it's computed later?
        # Actually _build_candidates in pipeline.py computes phash.
        
        for i in range(1, len(hashes)):
            dist = dedup.hamming_distance(hashes[i-1], hashes[i])
            ctx.observed_intra_track_distances.append(float(dist))

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
