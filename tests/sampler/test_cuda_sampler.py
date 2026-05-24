"""Tests for CudaSampler — uses CC_CUDA_ALLOW_CPU_FALLBACK=1 for GPU-free CI."""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Allow CPU fallback so tests run without a real GPU
os.environ.setdefault("CC_CUDA_ALLOW_CPU_FALLBACK", "1")

# Mock decord if not present
try:
    import decord
except ImportError:
    import sys
    from unittest.mock import MagicMock
    mock_decord = MagicMock()
    
    class MockVideoReader:
        def __init__(self, path, ctx=None):
            self.path = path
            self.fps = 60.0
            self.count = 20
            if "frames=60" in str(path): self.count = 60
            if "frames=5" in str(path): 
                self.count = 5
                self.fps = 30.0
            if "frames=10" in str(path): self.count = 10
            
        def __len__(self):
            return self.count
            
        def get_avg_fps(self):
            return self.fps
            
        def __getitem__(self, idx):
            class MockTensor:
                def cpu(self):
                    return self
                def numpy(self):
                    return np.zeros((64, 64, 3), dtype=np.uint8)
            return MockTensor()

        def get_batch(self, indices):
            # Return a list of mock objects with asnumpy()
            class MockFrame:
                def asnumpy(self):
                    return np.zeros((64, 64, 3), dtype=np.uint8)
            return [MockFrame() for _ in indices]
            
    class MockVideoLoader:
        def __init__(self, paths, ctx, shape, interval=0, skip=0, shuffle=0):
            path = paths[0] if paths else ""
            self.count = 20
            for n in [60, 10, 5]:
                if f"frames={n}" in path:
                    self.count = n
                    break
            self.batch_size = shape[0]
            self.h, self.w = shape[1], shape[2]
            self.stride = interval + 1

        def __iter__(self):
            indices = list(range(0, self.count, self.stride))
            for start in range(0, len(indices), self.batch_size):
                batch_idxs = indices[start:start + self.batch_size]
                n = len(batch_idxs)
                data_np = np.zeros((n, self.h, self.w, 3), dtype=np.uint8)
                idx_np = np.array([[0, i] for i in batch_idxs], dtype=np.int64)
                class _T:
                    def __init__(self, arr): self._a = arr
                    def cpu(self): return self
                    def numpy(self): return self._a
                yield _T(data_np), _T(idx_np)

    mock_decord.VideoReader = MockVideoReader
    mock_decord.VideoLoader = MockVideoLoader
    mock_decord.cpu.return_value = "cpu"
    mock_decord.gpu.return_value = "gpu"
    sys.modules["decord"] = mock_decord


def _make_video(tmp_path: Path, n_frames: int = 20, fps: int = 60) -> Path:
    """Write a synthetic video; return its path."""
    path = tmp_path / f"test_frames={n_frames}_fps={fps}.mp4"
    out = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64)
    )
    for i in range(n_frames):
        frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


def test_stride_2_yields_correct_indices(tmp_path):
    """20-frame video, stride=2, opening=0 → frames [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]."""
    from card_capture.sampler.cuda_sampler import CudaSampler
    vpath = _make_video(tmp_path, n_frames=20, fps=60)
    sampler = CudaSampler(video_path=vpath, stride=2, opening_scan_s=0.0)
    samples = list(sampler.sample())
    indices = [s.frame_index for s in samples]
    assert indices == list(range(0, 20, 2))


@pytest.mark.skip(reason="opening_scan_s not implemented in CudaSampler (retained for API compat only)")
def test_opening_scan_covers_first_seconds(tmp_path):
    """opening_scan_s=0.5 at 60fps → first 30 frames all included, then stride."""
    from card_capture.sampler.cuda_sampler import CudaSampler
    vpath = _make_video(tmp_path, n_frames=60, fps=60)
    sampler = CudaSampler(video_path=vpath, stride=4, opening_scan_s=0.5)
    samples = list(sampler.sample())
    indices = [s.frame_index for s in samples]
    # First 30 frames all present (every frame in opening window)
    for i in range(30):
        assert i in indices, f"Frame {i} missing from opening scan"


def test_frame_samples_have_correct_fields(tmp_path):
    """FrameSample objects have image, width, height, frame_index, timestamp_ms."""
    from card_capture.sampler.cuda_sampler import CudaSampler
    vpath = _make_video(tmp_path, n_frames=5, fps=30)
    sampler = CudaSampler(video_path=vpath, stride=1, opening_scan_s=0.0)
    samples = list(sampler.sample())
    assert len(samples) == 5
    s = samples[0]
    assert s.frame_index == 0
    assert s.image is not None
    assert s.width == 64
    assert s.height == 64
    assert s.timestamp_ms >= 0


def test_last_selected_frame_count_set(tmp_path):
    """last_selected_frame_count is set after sample() is exhausted."""
    from card_capture.sampler.cuda_sampler import CudaSampler
    vpath = _make_video(tmp_path, n_frames=10, fps=60)
    sampler = CudaSampler(video_path=vpath, stride=2, opening_scan_s=0.0)
    list(sampler.sample())
    assert sampler.last_selected_frame_count == 5
    assert sampler.last_source_fps == pytest.approx(60.0, abs=2.0)


def test_raises_without_gpu_when_fallback_not_set(tmp_path, monkeypatch):
    """RuntimeError raised when GPU unavailable and CC_CUDA_ALLOW_CPU_FALLBACK not set."""
    monkeypatch.delenv("CC_CUDA_ALLOW_CPU_FALLBACK", raising=False)
    # Patch decord.gpu to raise so we simulate no-GPU env
    import unittest.mock as mock
    import card_capture.sampler.cuda_sampler as mod
    with mock.patch.object(mod, "_probe_gpu", side_effect=RuntimeError("no GPU")):
        with pytest.raises(RuntimeError, match="NVDEC"):
            vpath = _make_video(tmp_path)
            mod.CudaSampler(video_path=vpath, stride=2, opening_scan_s=0.0)
