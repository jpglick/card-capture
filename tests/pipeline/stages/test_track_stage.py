"""Phase 4 — track stage emits V4-shape tracks_data dicts."""
from unittest.mock import MagicMock

import pytest

from card_capture.pipeline.stages import track as track_stage


def _detection(frame_index, det_id, confidence=0.9):
    return {
        "detection_id": det_id,
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 33,
        "width": 3840,
        "height": 2160,
        "corners": [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
        "confidence": confidence,
        "novelty_score": 1.0,
        "triage_metrics": {},
    }


def test_track_stage_writes_tracks_data_list_of_dicts():
    request = MagicMock()
    request.config = {"tracker_backend": "bytetrack", "min_track_length": 1}
    state = {
        "request": request,
        "sampled_frames": [],
        "novelty_scored_detections": [_detection(i, i) for i in range(5)],
    }
    track_stage.run(state, telemetry=MagicMock())
    assert "tracks_data" in state
    assert isinstance(state["tracks_data"], list)
    for t in state["tracks_data"]:
        assert "instance_id" in t and isinstance(t["instance_id"], str)
        assert "candidates" in t and isinstance(t["candidates"], list)
        for c in t["candidates"]:
            assert {"frame_index", "corners", "confidence",
                    "timestamp_ms", "width", "height",
                    "score_total", "detection_id"} <= set(c.keys())
