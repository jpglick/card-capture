"""Phase 6 — resolve stage groups sessions and predicts F/B."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.stages import resolve


def _refined(instance_id, q=0.8, side_score=0.5):
    img = (np.random.RandomState(0).rand(1050, 750, 3) * 255).astype(np.uint8)
    return {
        "instance_id": instance_id,
        "track_id": 1,
        "angle": "Unknown",
        "session_id": 0,
        "first_frame_index": 100,
        "best_canonical_image": img,
        "side_score": side_score,
        "appearance_vector": [0.1] * 320,
        "frame_entries": [{"quality_score": q, "visual_hash": "a" * 16, "is_canonical": True,
                           "candidate": MagicMock(detection_id=1, timestamp_ms=3333)}],
        "reid_embedding": [0.5, 0.5, 0.0, 0.0],
    }


def test_resolve_groups_tracks_by_session_id():
    request = MagicMock()
    request.config = {"use_fb_classifier": False}
    state = {
        "request": request,
        "refined_tracks": [
            {**_refined("t1"), "session_id": 10},
            {**_refined("t2"), "session_id": 10},
            {**_refined("t3"), "session_id": 20},
        ],
    }
    resolve.run(state, telemetry=MagicMock())
    assert len(state["resolved_sessions"]) == 2
    # session 10 has 2 tracks, session 20 has 1
    counts = [len(s["tracks"]) for s in state["resolved_sessions"]]
    assert sorted(counts) == [1, 2]


def test_resolve_assigns_front_to_highest_side_score():
    request = MagicMock()
    request.config = {"use_fb_classifier": False}
    state = {
        "request": request,
        "refined_tracks": [
            {**_refined("low-q"), "side_score": 0.1, "session_id": 1},
            {**_refined("high-q"), "side_score": 0.9, "session_id": 1},
        ],
    }
    resolve.run(state, telemetry=MagicMock())
    session = state["resolved_sessions"][0]
    # In V4/V5.5, the tracks are prioritized. The best is Front.
    best = session["tracks"][0]
    assert best["instance_id"] == "high-q"
    assert best["angle"] == "Front"


def test_resolve_classifier_can_override_heuristic(monkeypatch):
    """If classifier is 99% sure it's Back, it should override the side_score heuristic."""
    request = MagicMock()
    request.config = {"use_fb_classifier": True}
    state = {
        "request": request,
        "refined_tracks": [
            {**_refined("high-side"), "side_score": 0.9, "session_id": 1},
        ],
    }

    import card_capture.stages.resolve

    class _StubPredictor:
        def predict_array(self, img):
            # [Front_prob, Back_prob]
            return [0.01, 0.99]

    monkeypatch.setattr(resolve, "_get_fb_predictor", lambda: _StubPredictor())
    
    resolve.run(state, telemetry=MagicMock())
    assert state["resolved_sessions"][0]["tracks"][0]["angle"] == "Back"


def test_resolve_emits_prepared_tracks_list():
    """Output for fuse stage is a flat list of tracks with angle resolved."""
    request = MagicMock()
    request.config = {"use_fb_classifier": False}
    state = {
        "request": request,
        "refined_tracks": [_refined("a"), _refined("b")],
    }
    resolve.run(state, telemetry=MagicMock())
    assert "prepared_tracks" in state
    assert len(state["prepared_tracks"]) == 2
    assert {t["instance_id"] for t in state["prepared_tracks"]} == {"a", "b"}
    assert all(t["angle"] in ("Front", "Back") for t in state["prepared_tracks"])
