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

    monkeypatch.setattr("card_capture.pipeline_utils.decord", fake_decord, raising=False)

    from card_capture.pipeline_utils import decode_frames_gpu
    result = decode_frames_gpu("/fake/video.mov", [5, 2, 8])

    # Indices passed to get_batch must be sorted
    fake_vr.get_batch.assert_called_once_with([2, 5, 8])
    assert set(result.keys()) == {2, 5, 8}


def test_decode_frames_gpu_hard_fails_without_flag(monkeypatch):
    monkeypatch.delenv("CC_CUDA_ALLOW_CPU_FALLBACK", raising=False)

    fake_decord = MagicMock()
    fake_decord.gpu.side_effect = RuntimeError("no GPU")
    monkeypatch.setattr("card_capture.pipeline_utils.decord", fake_decord, raising=False)

    from card_capture.pipeline_utils import decode_frames_gpu
    with pytest.raises(RuntimeError, match="CC_CUDA_ALLOW_CPU_FALLBACK"):
        decode_frames_gpu("/fake/video.mov", [0, 1])


def test_decode_frames_gpu_cpu_fallback_with_flag(monkeypatch):
    monkeypatch.setenv("CC_CUDA_ALLOW_CPU_FALLBACK", "1")

    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_batch = MagicMock()
    fake_batch.__getitem__ = lambda self, i: MagicMock(asnumpy=lambda: fake_frame)
    fake_vr = MagicMock()
    fake_vr.get_batch.return_value = fake_batch

    fake_decord = MagicMock()
    fake_decord.gpu.side_effect = RuntimeError("no GPU")
    fake_decord.cpu.return_value = "cpu_ctx"
    fake_decord.VideoReader.return_value = fake_vr
    monkeypatch.setattr("card_capture.pipeline_utils.decord", fake_decord, raising=False)

    from card_capture.pipeline_utils import decode_frames_gpu
    result = decode_frames_gpu("/fake/video.mov", [3])
    fake_decord.cpu.assert_called_once_with(0)
    assert 3 in result


def test_decode_frames_gpu_empty_indices():
    from card_capture.pipeline_utils import decode_frames_gpu
    result = decode_frames_gpu("/fake/video.mov", [])
    assert result == {}


def test_compute_laplacian_scan_indices_basic():
    from card_capture.pipeline_utils import _compute_laplacian_scan_indices
    track_ranges = [
        {"instance_id": "a", "detections": [(10, []), (20, [])]},
    ]
    result = _compute_laplacian_scan_indices(track_ranges, scan_stride=5)
    # range(10, 21, 5) = {10, 15, 20}
    assert result == {10, 15, 20}


def test_compute_laplacian_scan_indices_multiple_tracks():
    from card_capture.pipeline_utils import _compute_laplacian_scan_indices
    track_ranges = [
        {"instance_id": "a", "detections": [(0, []), (4, [])]},
        {"instance_id": "b", "detections": [(10, []), (12, [])]},
    ]
    result = _compute_laplacian_scan_indices(track_ranges, scan_stride=2)
    assert {0, 2, 4} <= result
    assert {10, 12} <= result


def test_compute_laplacian_scan_indices_empty():
    from card_capture.pipeline_utils import _compute_laplacian_scan_indices
    assert _compute_laplacian_scan_indices([], scan_stride=4) == set()
    assert _compute_laplacian_scan_indices([{"instance_id": "a", "detections": []}], scan_stride=4) == set()
