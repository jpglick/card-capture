"""Tests for _laplacian_select_frames — in-track Laplacian quality scan."""
import numpy as np
import pytest


def _make_video(tmp_path, frames: list, fps: int = 30):
    """Write a synthetic video and return its path."""
    import cv2
    path = tmp_path / "test.mp4"
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in frames:
        out.write(f if f.ndim == 3 else np.stack([f] * 3, axis=-1))
    out.release()
    return path


def _blurry():
    """Uniform gray — zero Laplacian variance."""
    return np.full((64, 64, 3), 128, dtype=np.uint8)


def _sharp():
    """Checkerboard pattern — high Laplacian variance."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[::4, :] = 255
    img[:, ::4] = 255
    return img


CORNERS = [[0, 0], [63, 0], [63, 63], [0, 63]]


def test_selects_sharpest_frame(tmp_path):
    """Given blurry-sharp-blurry, frame 1 (sharp) is selected."""
    from card_capture.shared.pipeline_utils import _laplacian_select_frames
    vpath = _make_video(tmp_path, [_blurry(), _sharp(), _blurry()])
    track_ranges = [{
        "instance_id": "abc",
        "detections": [(0, CORNERS), (1, CORNERS), (2, CORNERS)],
    }]
    result = _laplacian_select_frames(vpath, track_ranges, scan_stride=1, top_k=1, max_corner_gap=5)
    assert "abc" in result
    assert len(result["abc"]) == 1
    assert result["abc"][0][0] == 1  # frame index 1 = sharp


def test_empty_track_ranges(tmp_path):
    """Empty input returns empty dict without errors."""
    from card_capture.shared.pipeline_utils import _laplacian_select_frames
    result = _laplacian_select_frames(tmp_path / "nonexistent.mp4", [], scan_stride=4, top_k=1, max_corner_gap=15)
    assert result == {}


def test_fallback_when_corner_gap_exceeded(tmp_path):
    """Sharpest frame far from any detection falls back to nearest detection."""
    from card_capture.shared.pipeline_utils import _laplacian_select_frames
    # 5 blurry + 1 sharp at frame 5; detection only at frame 0 (gap=5 > max=4)
    frames = [_blurry()] * 5 + [_sharp()]
    vpath = _make_video(tmp_path, frames)
    track_ranges = [{
        "instance_id": "xyz",
        "detections": [(0, CORNERS)],
    }]
    result = _laplacian_select_frames(vpath, track_ranges, scan_stride=1, top_k=1, max_corner_gap=4)
    assert result["xyz"][0][0] == 0  # falls back to nearest detection


def test_borrows_corners_from_nearest_detection(tmp_path):
    """Non-YOLO sharp frame borrows corners from nearest detection."""
    from card_capture.shared.pipeline_utils import _laplacian_select_frames
    alt_corners = [[1, 1], [62, 1], [62, 62], [1, 62]]
    frames = [_blurry(), _sharp(), _blurry()]
    vpath = _make_video(tmp_path, frames)
    track_ranges = [{
        "instance_id": "abc",
        "detections": [(0, CORNERS), (2, alt_corners)],
    }]
    result = _laplacian_select_frames(vpath, track_ranges, scan_stride=1, top_k=1, max_corner_gap=5)
    fi, corners = result["abc"][0]
    assert fi == 1
    assert corners == CORNERS or corners == alt_corners


def test_top_k_returns_multiple_frames(tmp_path):
    """top_k=2 returns two frames ordered sharpest first."""
    from card_capture.shared.pipeline_utils import _laplacian_select_frames
    frames = [_blurry(), _sharp(), _blurry(), _sharp()]
    vpath = _make_video(tmp_path, frames)
    track_ranges = [{
        "instance_id": "abc",
        "detections": [(i, CORNERS) for i in range(4)],
    }]
    result = _laplacian_select_frames(vpath, track_ranges, scan_stride=1, top_k=2, max_corner_gap=5)
    assert len(result["abc"]) == 2
    sharp_frames = {result["abc"][0][0], result["abc"][1][0]}
    assert 1 in sharp_frames and 3 in sharp_frames
