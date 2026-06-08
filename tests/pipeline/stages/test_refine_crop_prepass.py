import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.core.config import PipelineConfig
from card_capture.stages import refine
from card_capture.stages.refine.cropper import PrecisionNormalizer


def test_config_exposes_crop_knobs_in_request_dict():
    cfg = PipelineConfig()
    d = cfg.to_request_config()
    assert d["refine_crop_margin_px"] == 8
    assert d["refine_min_available_mb"] == 2048.0


def test_quad_bbox_clamps_to_frame_and_applies_margin():
    # quad inside a 200x300 (w x h) frame; margin expands, clamped at 0/edge.
    corners = [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)]
    assert refine._quad_bbox(corners, 200, 300, margin=8) == (30, 42, 168, 248)


def test_quad_bbox_clamps_when_quad_touches_edge():
    corners = [(2.0, 1.0), (199.0, 0.0), (198.0, 299.0), (1.0, 298.0)]
    assert refine._quad_bbox(corners, 200, 300, margin=8) == (0, 0, 200, 300)


def test_crop_and_rebase_returns_a_copy_not_a_view():
    frame = (np.random.RandomState(1).rand(300, 200, 3) * 255).astype(np.uint8)
    corners = [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)]
    crop, _ = refine._crop_and_rebase(frame, corners, margin=8)
    assert crop.base is None  # a real copy, so freeing `frame` releases its memory


def test_crop_and_rebase_warp_equivalence():
    """warp(full, corners) is pixel-identical to warp(crop, rebased corners)."""
    frame = (np.random.RandomState(0).rand(300, 200, 3) * 255).astype(np.uint8)
    corners = [(40.0, 50.0), (160.0, 55.0), (158.0, 240.0), (38.0, 235.0)]
    norm = PrecisionNormalizer()

    full = norm.normalize(frame, corners, rotate_180=False)
    crop, local = refine._crop_and_rebase(frame, corners, margin=8)
    cropped = norm.normalize(crop, local, rotate_180=False)

    assert np.array_equal(full, cropped)


def _candidate(detection_id, frame_index, score):
    return {
        "detection_id": detection_id,
        "frame_index": frame_index,
        "score_total": score,
        "corners": [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    }


def test_plan_crop_candidates_keeps_top_n_per_track_keyed_by_detection():
    tracks = [
        {"candidates": [_candidate(1, 5, 0.9), _candidate(2, 6, 0.1), _candidate(3, 7, 0.5)]},
    ]
    wanted = refine._plan_crop_candidates(tracks, top_n=2)
    # Top-2 by score = detections 1 (0.9) and 3 (0.5); detection 2 dropped.
    assert set(wanted.keys()) == {1, 3}
    assert wanted[1] == (5, [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
