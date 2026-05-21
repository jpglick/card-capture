"""Tests for CudaSampler VideoLoader-based implementation."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_batch(n, h=100, w=100):
    """Return a fake VideoLoader (batch_data, batch_indices) pair."""
    data = MagicMock()
    data.asnumpy.return_value = np.zeros((n, h, w, 3), dtype=np.uint8)
    indices = MagicMock()
    indices.asnumpy.return_value = np.arange(n * 2, step=2).reshape(-1)
    return data, indices


def _make_probe_vr(total=60, fps=30.0, h=100, w=100):
    first_frame = MagicMock()
    first_frame.asnumpy.return_value = np.zeros((h, w, 3), dtype=np.uint8)
    vr = MagicMock()
    vr.__len__ = lambda self: total
    vr.get_avg_fps.return_value = fps
    vr.__getitem__ = lambda self, i: first_frame
    return vr


@patch("card_capture.sampler.cuda_sampler.decord")
def test_sample_batches_uses_video_loader(mock_decord):
    mock_decord.gpu.return_value = "gpu_ctx"
    mock_decord.cpu.return_value = "cpu_ctx"

    probe_vr = _make_probe_vr(total=60, fps=30.0, h=100, w=100)
    mock_decord.VideoReader.return_value = probe_vr

    batch1 = _make_batch(4)
    mock_vl = MagicMock()
    mock_vl.__iter__ = MagicMock(return_value=iter([batch1]))
    mock_decord.VideoLoader.return_value = mock_vl

    from card_capture.sampler.cuda_sampler import CudaSampler
    sampler = CudaSampler(stride=2)
    batches = list(sampler.sample_batches(batch_size=4, video_path="/fake/v.mov"))

    # VideoLoader must be constructed — interval=stride-1=1 for stride=2
    mock_decord.VideoLoader.assert_called_once()
    call_kwargs = mock_decord.VideoLoader.call_args[1]
    assert call_kwargs["interval"] == 1

    assert len(batches) == 1
    assert len(batches[0]) == 4
    from card_capture.models import FrameSample
    assert isinstance(batches[0][0], FrameSample)


@patch("card_capture.sampler.cuda_sampler.decord")
def test_sample_batches_hard_fails_without_gpu(mock_decord, monkeypatch):
    monkeypatch.delenv("CC_CUDA_ALLOW_CPU_FALLBACK", raising=False)
    mock_decord.gpu.side_effect = RuntimeError("no GPU")

    from card_capture.sampler.cuda_sampler import CudaSampler
    with pytest.raises(RuntimeError, match="CC_CUDA_ALLOW_CPU_FALLBACK"):
        CudaSampler(stride=2)


@patch("card_capture.sampler.cuda_sampler.decord")
def test_sample_yields_frame_samples(mock_decord):
    mock_decord.gpu.return_value = "gpu_ctx"
    mock_decord.cpu.return_value = "cpu_ctx"

    probe_vr = _make_probe_vr(total=4, fps=30.0, h=50, w=80)
    mock_decord.VideoReader.return_value = probe_vr

    batch1 = _make_batch(2, h=50, w=80)
    mock_vl = MagicMock()
    mock_vl.__iter__ = MagicMock(return_value=iter([batch1]))
    mock_decord.VideoLoader.return_value = mock_vl

    from card_capture.sampler.cuda_sampler import CudaSampler
    from card_capture.models import FrameSample
    sampler = CudaSampler(stride=2)
    frames = list(sampler.sample(video_path="/fake/v.mov"))

    assert all(isinstance(f, FrameSample) for f in frames)
    assert all(f.width == 80 and f.height == 50 for f in frames)


@patch("card_capture.sampler.cuda_sampler.decord")
def test_stride_1_uses_interval_0(mock_decord):
    mock_decord.gpu.return_value = "gpu_ctx"
    probe_vr = _make_probe_vr(total=10, fps=30.0)
    mock_decord.VideoReader.return_value = probe_vr
    mock_vl = MagicMock()
    mock_vl.__iter__ = MagicMock(return_value=iter([]))
    mock_decord.VideoLoader.return_value = mock_vl

    from card_capture.sampler.cuda_sampler import CudaSampler
    sampler = CudaSampler(stride=1)
    list(sampler.sample_batches(batch_size=4, video_path="/fake/v.mov"))

    call_kwargs = mock_decord.VideoLoader.call_args[1]
    assert call_kwargs["interval"] == 0
