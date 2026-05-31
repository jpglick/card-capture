"""Stage 5: Session-Aware Tracking.

Tracker backend (BoT-SORT or ByteTrack) is selected from request.config.
We emit both ``state["tracks"]`` (List[TrackState], legacy) and
``state["tracks_data"]`` (List[Dict] V4 shape) so the back-half stages
can consume the richer per-candidate dicts the V4 refine step expects.
"""
from __future__ import annotations

from card_capture.pipeline.stage_metrics import emit_stage_metrics
from card_capture.tracking.botsort_adapter import BoTSORTAdapter
from card_capture.tracking.bytetrack_adapter import ByteTrackAdapter


def run(state: dict, *, telemetry) -> None:
    cfg = state["request"].config
    backend = cfg.get("tracker_backend", "botsort")
    
    # Common tracker args - extracted from cfg to match adapter signatures
    kwargs = {
        "min_track_length": cfg.get("min_track_length", 3),
        "track_activation_threshold": cfg.get("track_activation_threshold", 0.25),
        "lost_track_buffer": cfg.get("lost_track_buffer", 30),
        "minimum_matching_threshold": cfg.get("minimum_matching_threshold", 0.8),
    }

    if backend == "botsort":
        kwargs["centroid_jump_ratio"] = cfg.get("centroid_jump_ratio", 0.30)
        kwargs["centroid_jump_frames"] = cfg.get("centroid_jump_frames", 3)
        # Appearance sessionization settings
        kwargs["same_threshold"] = cfg.get("appearance_same_threshold", 0.15)
        kwargs["change_threshold"] = cfg.get("appearance_change_threshold", 0.30)
        kwargs["confirm_frames"] = cfg.get("appearance_confirm_frames", 3)
        kwargs["bridge_min_occurrences"] = cfg.get("bridge_min_occurrences", 3)
        kwargs["bridge_position_ratio"] = cfg.get("bridge_position_ratio", 0.80)
        kwargs["bridge_neighbor_change_ratio"] = cfg.get("bridge_neighbor_change_ratio", 0.80)
        kwargs["bridge_novelty_margin"] = cfg.get("bridge_novelty_margin", 0.05)
        kwargs["bridge_max_length_ratio"] = cfg.get("bridge_max_length_ratio", 0.75)
        
        tracker = BoTSORTAdapter(**kwargs)
    else:
        tracker = ByteTrackAdapter(**kwargs)

    detections = state["novelty_scored_detections"]
    frames = state["sampled_frames"]
    
    runtime_worker = state.get("runtime_worker")
    if runtime_worker is None:
        track_states = tracker.assign(detections, frames)
    else:
        track_states = runtime_worker.call(tracker.assign, detections, frames)

    state["tracks"] = track_states

    # Build V4-shape tracks_data: List[Dict] with rich per-candidate dicts.
    # Each ScoredCandidate is enriched with the originating detection row's
    # width/height/confidence/timestamp/corners so refine can reuse them
    # without re-decoding.
    by_det_id = {d["detection_id"]: d for d in detections}
    tracks_data: list[dict] = []
    for ts in track_states:
        candidates: list[dict] = []
        for sc in ts.candidates:
            det = by_det_id.get(sc.detection_id, {})
            candidates.append({
                "detection_id": sc.detection_id,
                "frame_index": sc.frame_index,
                "timestamp_ms": int(sc.timestamp_ms),
                "width": int(det.get("width", 0)),
                "height": int(det.get("height", 0)),
                "corners": [(float(x), float(y)) for x, y in (det.get("corners") or [])],
                "confidence": float(det.get("confidence", 0.0)),
                "novelty_score": float(det.get("novelty_score", 1.0)),
                "score_total": float(getattr(sc.score, "total", 0.0)),
                "image_path": "",
                "triage_metrics": det.get("triage_metrics", {}),
            })
        tracks_data.append({
            "instance_id": ts.instance_id,
            "track_id": int(getattr(ts, "track_id", 0) or 0),
            "angle": "Unknown",
            "session_id": int(getattr(ts, "session_id", 0) or 0),
            "first_frame_index": int(getattr(ts, "first_frame_index", -1) or -1),
            "candidates": candidates,
            "reid_embedding": getattr(ts, "reid_embedding", None),
        })
    state["tracks_data"] = tracks_data
    metrics = {"tracks_final": len(track_states), "tracks_data": len(tracks_data)}
    metrics.update(getattr(tracker, "sessionization_metrics", {}))
    emit_stage_metrics(state, stage="track", metrics=metrics)
