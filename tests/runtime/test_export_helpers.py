from __future__ import annotations

import numpy as np
import pytest
import torch

from card_capture.runtime.batches import (
    GpuCropBatch,
    GpuEmbeddingBatch,
    to_cpu_for_score,
    to_cpu_for_phash,
    to_cpu_for_dedup,
    to_cpu_for_fuse,
    to_cpu_for_export,
)


def _gpu_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("no GPU available")


def test_to_cpu_for_score_returns_dict():
    dev = _gpu_device()
    crops = GpuCropBatch(tensor=torch.rand((2, 3, 1050, 750), device=dev))
    scores = to_cpu_for_score(crops)
    assert isinstance(scores, list)
    assert all(isinstance(s, dict) for s in scores)
    for s in scores:
        # Required keys per spec Section 2 table
        for k in ("sharpness", "glare", "aspect_ratio", "size", "complexity", "border_purity", "confidence"):
            assert k in s


def test_to_cpu_for_phash_returns_hex():
    pytest.importorskip("imagehash")
    dev = _gpu_device()
    crops = GpuCropBatch(tensor=torch.rand((2, 3, 1050, 750), device=dev))
    hashes = to_cpu_for_phash(crops)
    assert len(hashes) == 2
    assert all(isinstance(h, str) for h in hashes)
    assert all(int(h, 16) >= 0 for h in hashes)


def test_to_cpu_for_dedup_returns_float32_arrays():
    dev = _gpu_device()
    emb = GpuEmbeddingBatch(tensor=torch.rand((3, 384), device=dev))
    arrs = to_cpu_for_dedup(emb)
    assert isinstance(arrs, np.ndarray)
    assert arrs.dtype == np.float32
    assert arrs.shape == (3, 384)


def test_to_cpu_for_export_writes_png(tmp_path):
    dev = _gpu_device()
    crops = GpuCropBatch(tensor=torch.rand((1, 3, 1050, 750), device=dev))
    paths = to_cpu_for_export(crops, out_dir=tmp_path, basenames=["card_0"])
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".png"
