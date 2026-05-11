import numpy as np
import pytest

from card_capture.presence.background_novelty import (
    BackgroundModel,
    quad_novelty,
    is_quad_card_like,
)


def _solid(h, w, value):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_background_model_from_frames_averages_grayscale():
    frames = [_solid(40, 50, 100), _solid(40, 50, 120)]
    bg = BackgroundModel.from_frames(frames)
    assert bg.gray.shape == (40, 50)
    assert pytest.approx(float(bg.gray.mean()), abs=1.0) == 110.0


def test_quad_novelty_zero_for_background_patch():
    """A quad whose interior pixels equal the background returns ~0 novelty."""
    bg_frame = _solid(200, 200, 128)
    bg = BackgroundModel.from_frames([bg_frame] * 5)
    full_frame = _solid(200, 200, 128)
    corners = [(40.0, 40.0), (160.0, 40.0), (160.0, 160.0), (40.0, 160.0)]
    score = quad_novelty(full_frame, corners, bg)
    assert score < 0.05, f"expected near-zero novelty, got {score}"


def test_quad_novelty_high_for_painted_card():
    """A quad whose interior differs strongly from background returns high novelty."""
    bg_frame = _solid(200, 200, 128)
    bg = BackgroundModel.from_frames([bg_frame] * 5)
    # Paint a high-contrast rectangle inside the same coordinates.
    frame = bg_frame.copy()
    frame[40:160, 40:160] = 20  # very different from 128
    corners = [(40.0, 40.0), (160.0, 40.0), (160.0, 160.0), (40.0, 160.0)]
    score = quad_novelty(frame, corners, bg)
    assert score > 0.30, f"expected high novelty, got {score}"


def test_is_quad_card_like_threshold():
    """is_quad_card_like applies the threshold."""
    bg = BackgroundModel.from_frames([_solid(200, 200, 128)])
    # Identical → reject
    assert not is_quad_card_like(_solid(200, 200, 128),
                                 [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
                                 bg, threshold=0.08)
    # Painted → accept
    frame = _solid(200, 200, 128)
    frame[10:50, 10:50] = 20
    assert is_quad_card_like(frame,
                             [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
                             bg, threshold=0.08)


def test_quad_novelty_handles_size_mismatch():
    """If the frame is a different size than the bg model, novelty resizes the bg slice
    rather than crashing."""
    bg = BackgroundModel.from_frames([_solid(100, 100, 128)])
    larger = _solid(400, 400, 200)  # different size AND different content
    corners = [(50.0, 50.0), (350.0, 50.0), (350.0, 350.0), (50.0, 350.0)]
    score = quad_novelty(larger, corners, bg)
    assert score > 0.20
