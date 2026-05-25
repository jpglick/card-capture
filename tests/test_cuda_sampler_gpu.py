"""sample_gpu_batches keeps the 4K tensor on-device and emits small thumbnails."""
import numpy as np
import pytest
from unittest.mock import MagicMock

torch = pytest.importorskip("torch")


def test_sample_gpu_batches_keeps_tensor_and_resizes_thumbnail(monkeypatch):
    import card_capture.sampler.cuda_sampler as cs

    H, W = 1080, 1920
    batch_tensor = torch.randint(0, 256, (2, H, W, 3), dtype=torch.uint8)
    indices = torch.tensor([[0, 4], [0, 6]])  # decord (video_idx, frame_idx) pairs

    fake_vr = MagicMock()
    fake_vr.__len__ = lambda self: 100
    fake_vr.get_avg_fps.return_value = 30.0
    fake_vr.__getitem__ = lambda self, i: torch.zeros((H, W, 3), dtype=torch.uint8)

    fake_loader = [(batch_tensor, indices)]

    fake_decord = MagicMock()
    fake_decord.cpu.return_value = "cpu_ctx"
    fake_decord.VideoReader.return_value = fake_vr
    fake_decord.VideoLoader.return_value = fake_loader
    monkeypatch.setattr(cs, "decord", fake_decord)

    sampler = cs.CudaSampler.__new__(cs.CudaSampler)
    sampler.video_path = "/fake/video.MOV"
    sampler.stride = 2
    sampler._gpu_ctx = "gpu_ctx"

    out = list(sampler.sample_gpu_batches(batch_size=2, thumbnail_width=640, video_path="/fake/video.MOV"))
    assert len(out) == 1
    gpu_batch, frames = out[0]

    assert isinstance(gpu_batch, torch.Tensor)
    assert gpu_batch.shape == (2, H, W, 3)

    assert len(frames) == 2
    for f in frames:
        assert f.width == W and f.height == H            # original dims for coord scale-back
        assert f.image.shape[1] == 640                   # thumbnail width
        assert f.image.shape[0] == round(H * 640 / W)    # aspect-preserved height
    assert [f.frame_index for f in frames] == [4, 6]
