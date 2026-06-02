"""Tests for GPU frame decode helpers in pipeline_utils."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_decode_frames_gpu_returns_index_map(monkeypatch):
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_batch = MagicMock()
    fake_batch.__getitem__ = lambda self, i: MagicMock(asnumpy=lambda: fake_frame)

    fake_vr = MagicMock()
    fake_vr.get_batch.return_value = fake_batch

    fake_decord = MagicMock()
    fake_decord.gpu.return_value = "gpu_ctx"
    fake_decord.VideoReader.return_value = fake_vr

    monkeypatch.setattr("card_capture.shared.pipeline_utils.decord", fake_decord, raising=False)

    from card_capture.shared.pipeline_utils import decode_frames_gpu
    result = decode_frames_gpu("/fake/video.mov", [5, 2, 8])

    # Indices passed to get_batch must be sorted
    fake_vr.get_batch.assert_called_once_with([2, 5, 8])
    assert set(result.keys()) == {2, 5, 8}


def test_decode_frames_gpu_uses_opencv_on_mac(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("card_capture.shared.pipeline_utils._decode_frames_opencv", return_value={0: fake_frame}) as mock_cv:
        from card_capture.shared.pipeline_utils import decode_frames_gpu
        res = decode_frames_gpu("vid.mov", [0])
        assert res[0].shape == (100, 100, 3)
        mock_cv.assert_called_once()


def test_decode_frames_gpu_empty_indices():
    from card_capture.shared.pipeline_utils import decode_frames_gpu
    result = decode_frames_gpu("/fake/video.mov", [])
    assert result == {}


def test_compute_laplacian_scan_indices_basic():
    from card_capture.shared.pipeline_utils import _compute_laplacian_scan_indices
    track_ranges = [
        {"instance_id": "a", "detections": [(10, []), (20, [])]},
    ]
    result = _compute_laplacian_scan_indices(track_ranges, scan_stride=5)
    # range(10, 21, 5) = {10, 15, 20}
    assert result == {10, 15, 20}


def test_compute_laplacian_scan_indices_multiple_tracks():
    from card_capture.shared.pipeline_utils import _compute_laplacian_scan_indices
    track_ranges = [
        {"instance_id": "a", "detections": [(0, []), (4, [])]},
        {"instance_id": "b", "detections": [(10, []), (12, [])]},
    ]
    result = _compute_laplacian_scan_indices(track_ranges, scan_stride=2)
    assert {0, 2, 4} <= result
    assert {10, 12} <= result


def test_compute_laplacian_scan_indices_empty():
    from card_capture.shared.pipeline_utils import _compute_laplacian_scan_indices
    assert _compute_laplacian_scan_indices([], scan_stride=4) == set()
    assert _compute_laplacian_scan_indices([{"instance_id": "a", "detections": []}], scan_stride=4) == set()


def test_laplacian_select_frames_uses_decoded_dict(monkeypatch):
    """When decoded_frames is provided, VideoCapture must NOT be opened."""
    import numpy as np
    from card_capture.shared.pipeline_utils import _laplacian_select_frames

    # Create a frame with some variance so Laplacian > 0
    sharp = np.zeros((100, 100, 3), dtype=np.uint8)
    sharp[40:60, 40:60] = 255

    decoded = {5: sharp, 6: sharp, 7: sharp}

    track_ranges = [
        {"instance_id": "t1", "detections": [
            (5, [[0, 0], [1, 0], [1, 1], [0, 1]]),
            (7, [[0, 0], [1, 0], [1, 1], [0, 1]]),
        ]},
    ]

    opened = []

    def fake_open_capture(p):
        opened.append(p)
        raise AssertionError("VideoCapture opened unexpectedly")

    monkeypatch.setattr("card_capture.shared.pipeline_utils._open_capture", fake_open_capture)

    result = _laplacian_select_frames(
        "/fake/video.mov", track_ranges,
        scan_stride=1, top_k=1, max_corner_gap=15,
        decoded_frames=decoded,
    )
    assert not opened, "VideoCapture should not have been opened"
    assert "t1" in result
    assert len(result["t1"]) == 1

def test_laplacian_variance_batch_cpu_fallback(monkeypatch):
    """When MPS is unavailable, _laplacian_variance_batch falls back to cv2 per-image."""
    import torch
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    from card_capture.shared.pipeline_utils import _laplacian_variance_batch

    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    results = _laplacian_variance_batch(images)

    assert len(results) == 3
    assert all(isinstance(r, float) for r in results)

def test_laplacian_variance_batch_returns_floats():
    """_laplacian_variance_batch always returns a list of floats, one per image."""
    from card_capture.shared.pipeline_utils import _laplacian_variance_batch

    images = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
    results = _laplacian_variance_batch(images)

    assert len(results) == 4
    assert all(isinstance(r, float) for r in results)
    assert all(r >= 0.0 for r in results)
