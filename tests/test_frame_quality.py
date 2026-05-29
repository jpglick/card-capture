# tests/test_frame_quality.py
import math
import numpy as np
import cv2
from card_capture.frame_quality import (
    flatness_score, clarity_score_gpu, clarity_var_gpu_roi,
    appearance_grid_gpu_roi, ahash_from_grid, ahash_hamming,
)


def test_perfect_5x7_rectangle_scores_near_one():
    # Axis-aligned rectangle with the card's 5:7 short:long ratio (500x700).
    corners = [(0.0, 0.0), (500.0, 0.0), (500.0, 700.0), (0.0, 700.0)]
    assert flatness_score(corners) > 0.95


def test_keystoned_quad_scores_lower_than_rectangle():
    rect = [(0.0, 0.0), (500.0, 0.0), (500.0, 700.0), (0.0, 700.0)]
    # Top edge much narrower than bottom — strong perspective skew.
    keystone = [(150.0, 0.0), (350.0, 0.0), (500.0, 700.0), (0.0, 700.0)]
    assert flatness_score(keystone) < flatness_score(rect)
    assert flatness_score(keystone) < 0.8


def test_corner_order_independent():
    rect = [(0.0, 0.0), (500.0, 0.0), (500.0, 700.0), (0.0, 700.0)]
    shuffled = [rect[2], rect[0], rect[3], rect[1]]
    assert abs(flatness_score(rect) - flatness_score(shuffled)) < 1e-6


def test_returns_zero_for_degenerate_input():
    assert flatness_score([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]) == 0.0
    assert flatness_score([(0.0, 0.0)]) == 0.0


def _checkerboard(h=200, w=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[::8, :, :] = 255
    img[:, ::8, :] = 255
    return img


def test_sharp_scores_higher_than_blurred_cpu_device():
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=4)
    bbox = (0, 0, 200, 200)  # x0, y0, x1, y1
    s_sharp = clarity_score_gpu(sharp, bbox, device="cpu")
    s_blur = clarity_score_gpu(blurred, bbox, device="cpu")
    assert s_sharp > s_blur
    assert s_blur >= 0.0


def test_clarity_uses_only_the_roi():
    # Sharp texture only inside the ROI; flat elsewhere → ROI score stays high.
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = _checkerboard(100, 100)
    full = clarity_score_gpu(img, (0, 0, 200, 200), device="cpu")
    roi = clarity_score_gpu(img, (50, 50, 150, 150), device="cpu")
    assert roi > full


def test_ahash_distinguishes_different_cards_but_not_same():
    import torch
    # Two visually different "cards" and one identical copy.
    rng = np.random.default_rng(0)
    card_a = rng.integers(0, 256, (400, 280, 3), dtype=np.uint8)
    card_b = rng.integers(0, 256, (400, 280, 3), dtype=np.uint8)
    bbox = (0, 0, 280, 400)
    ga = appearance_grid_gpu_roi(torch.from_numpy(card_a), bbox)
    gb = appearance_grid_gpu_roi(torch.from_numpy(card_b), bbox)
    ga2 = appearance_grid_gpu_roi(torch.from_numpy(card_a.copy()), bbox)
    ha, hb, ha2 = ahash_from_grid(ga), ahash_from_grid(gb), ahash_from_grid(ga2)
    assert ahash_hamming(ha, ha2) == 0            # identical content
    assert ahash_hamming(ha, hb) >= 18            # different cards exceed swap threshold


def test_ahash_stable_under_small_shift():
    import torch
    # Same card nudged a few pixels (camera jitter) stays below the swap threshold.
    base = np.zeros((400, 280, 3), dtype=np.uint8)
    base[80:320, 60:220] = _checkerboard(240, 160)
    shifted = np.zeros((400, 280, 3), dtype=np.uint8)
    shifted[83:323, 63:223] = _checkerboard(240, 160)
    bbox = (0, 0, 280, 400)
    h1 = ahash_from_grid(appearance_grid_gpu_roi(torch.from_numpy(base), bbox))
    h2 = ahash_from_grid(appearance_grid_gpu_roi(torch.from_numpy(shifted), bbox))
    assert ahash_hamming(h1, h2) < 18


def test_clarity_var_gpu_roi_ranks_sharp_over_blurred():
    # The on-device variant slices a GPU-resident frame tensor and returns a
    # 0-dim tensor (no sync). Ranking must match the numpy path: sharp > blurred.
    import torch
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=4)
    bbox = (0, 0, 200, 200)
    t_sharp = torch.from_numpy(sharp)  # HxWxC, BGR, on CPU device
    t_blur = torch.from_numpy(blurred)
    v_sharp = clarity_var_gpu_roi(t_sharp, bbox)
    v_blur = clarity_var_gpu_roi(t_blur, bbox)
    assert v_sharp.shape == ()  # 0-dim, stackable for a single batched sync
    assert float(v_sharp) > float(v_blur) >= 0.0
