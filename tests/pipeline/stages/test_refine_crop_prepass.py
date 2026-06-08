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
