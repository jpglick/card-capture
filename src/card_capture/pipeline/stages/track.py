"""Stage 5: Session-Aware Tracking.

Tracker backend (BoT-SORT or ByteTrack) is selected from request.config.
"""
from __future__ import annotations

from card_capture.tracking.botsort_adapter import BoTSORTAdapter
from card_capture.tracking.bytetrack_adapter import ByteTrackAdapter


def run(state: dict, *, telemetry) -> None:
    cfg = state["request"].config
    backend = cfg.get("tracker_backend", "bytetrack")
    if backend == "botsort":
        tracker = BoTSORTAdapter(cfg)
    else:
        tracker = ByteTrackAdapter(cfg)
    state["tracks"] = tracker.assign(state["novelty_scored_detections"], state["sampled_frames"])
