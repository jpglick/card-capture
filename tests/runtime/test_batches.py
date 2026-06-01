from __future__ import annotations

import pytest
import torch

from card_capture.runtime.batches import (
    GpuFrameBatch,
    GpuCropBatch,
    GpuEmbeddingBatch,
    WrongDeviceError,
)


def _gpu_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("no GPU available")


def test_frame_batch_accepts_gpu_tensor():
    dev = _gpu_device()
    t = torch.zeros((2, 3, 1080, 1920), device=dev, dtype=torch.float32)
    batch = GpuFrameBatch(tensor=t)
    assert batch.tensor.device.type == "mps"


def test_frame_batch_rejects_cpu_tensor():
    t = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    with pytest.raises(WrongDeviceError):
        GpuFrameBatch(tensor=t)


def test_crop_batch_enforces_layout():
    dev = _gpu_device()
    bad = torch.zeros((4, 750, 1050, 3), device=dev)  # NHWC instead of NCHW
    with pytest.raises(ValueError):
        GpuCropBatch(tensor=bad)


def test_embedding_batch_2d_only():
    dev = _gpu_device()
    e = torch.zeros((8, 384), device=dev)
    GpuEmbeddingBatch(tensor=e)  # ok
    with pytest.raises(ValueError):
        GpuEmbeddingBatch(tensor=torch.zeros((8, 384, 1), device=dev))
