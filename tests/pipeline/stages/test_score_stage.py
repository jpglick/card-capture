"""Phase 5 — score stage applies novelty / confidence / stand gates."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.pipeline.stages import score as score_stage


def _track(instance_id, frame_count, novelty=1.0, q=0.7, sharpness=0.7):
    return {
        "instance_id": instance_id,
        "frame_entries": [
            {
                "novelty_score": novelty,
                "quality_score": q,
                "score_total": q,
                "confidence": q,
                "quality_components": {"sharpness": sharpness},
            }
            for _ in range(frame_count)
        ],
    }


def test_score_passes_through_when_no_gates_active():
    """All gates off → no track pruned, scored_tracks shape preserved."""
    request = MagicMock()
    request.config = {
        "novelty_floor": 0.0,
        "track_confidence_floor": 0.0,
        "stand_novelty_max": 0.0,
        "stand_sharpness_max": 0.0,
    }
    state = {
        "request": request,
        "refined_tracks": [
            _track("a", 5, novelty=1.0, q=0.9, sharpness=0.9),
            _track("b", 5, novelty=0.05, q=0.4, sharpness=0.1),
        ],
        "bg_model": None,
    }
    score_stage.run(state, telemetry=MagicMock())
    assert len(state["scored_tracks"]) == 2
    assert all(not t["pruned"] for t in state["scored_tracks"])
    assert state["pruned_instance_ids"] == []


def test_score_confidence_floor_prunes_low_quality():
    request = MagicMock()
    request.config = {
        "novelty_floor": 0.0,
        "track_confidence_floor": 0.60,
        "stand_novelty_max": 0.0,
        "stand_sharpness_max": 0.0,
    }
    state = {
        "request": request,
        "refined_tracks": [
            _track("strong", 5, q=0.8),
            _track("weak", 5, q=0.45),
        ],
        "bg_model": None,
    }
    score_stage.run(state, telemetry=MagicMock())
    assert "weak" in state["pruned_instance_ids"]
    assert "strong" not in state["pruned_instance_ids"]


def test_score_novelty_gate_useful_requires_n5_std015_min035():
    """Gate stays off when there's no useful spread."""
    from card_capture.pipeline.stages.score import _novelty_gate_useful
    assert _novelty_gate_useful([1.0] * 10) is False  # std 0
    assert _novelty_gate_useful([0.5, 0.51, 0.52]) is False  # n < 5
    # Wide spread, low min → useful
    assert _novelty_gate_useful([0.1, 0.2, 0.7, 0.8, 0.9]) is True


def test_score_adaptive_novelty_threshold_is_largest_gap_midpoint():
    """Two-cluster novelty distribution → threshold lands in the gap."""
    request = MagicMock()
    request.config = {
        "novelty_floor": 0.30,
        "track_confidence_floor": 0.0,
        "stand_novelty_max": 0.0,
        "stand_sharpness_max": 0.0,
    }
    # Background model present, two real cards at high novelty, two phantoms at low
    state = {
        "request": request,
        "refined_tracks": [
            _track("real-1", 6, novelty=0.85, q=0.8),
            _track("real-2", 6, novelty=0.80, q=0.8),
            _track("phantom-1", 6, novelty=0.20, q=0.8),
            _track("phantom-2", 6, novelty=0.15, q=0.8),
            _track("real-3", 6, novelty=0.82, q=0.8),
        ],
        "bg_model": object(),  # truthy sentinel
    }
    score_stage.run(state, telemetry=MagicMock())
    pruned = set(state["pruned_instance_ids"])
    assert "phantom-1" in pruned and "phantom-2" in pruned
    assert "real-1" not in pruned and "real-2" not in pruned


def test_score_stand_gate_prunes_low_novelty_and_low_sharpness():
    """Stand gate active → prunes tracks that are both dull and low-novelty."""
    request = MagicMock()
    request.config = {
        "novelty_floor": 0.0,
        "track_confidence_floor": 0.0,
        "stand_novelty_max": 0.40,
        "stand_sharpness_max": 0.30,
    }
    state = {
        "request": request,
        "refined_tracks": [
            _track("stand", 5, novelty=0.2, sharpness=0.1),  # Both low → prune
            _track("shiny", 5, novelty=0.2, sharpness=0.8),  # Low nov, high sharp → keep
            _track("card", 5, novelty=0.9, sharpness=0.1),   # High nov, low sharp → keep
        ],
        "bg_model": object(),
    }
    score_stage.run(state, telemetry=MagicMock())
    pruned = state["pruned_instance_ids"]
    assert "stand" in pruned
    assert "shiny" not in pruned
    assert "card" not in pruned
