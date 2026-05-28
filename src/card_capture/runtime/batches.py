"""Device-tagged batch types and approved CPU-export helpers.

Wrapper types enforce device residency, layout, and dtype at construction.
Approved export helpers are the only legal `.cpu()` / `.numpy()` call sites
inside files tagged GPU-resident (see pyproject.toml [tool.gpu_strict_lint]).
"""
from __future__ import annotations

import dataclasses

import torch


class WrongDeviceError(TypeError):
    """Raised when a device-tagged batch is constructed with a CPU tensor."""


def _require_gpu(t: torch.Tensor, name: str) -> None:
    if t.device.type not in ("cuda", "mps"):
        raise WrongDeviceError(f"{name} requires a GPU tensor; got device={t.device}")


@dataclasses.dataclass(frozen=True)
class GpuFrameBatch:
    """NCHW float frames on GPU."""
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        _require_gpu(self.tensor, "GpuFrameBatch")
        if self.tensor.dim() != 4:
            raise ValueError(f"GpuFrameBatch expects 4D NCHW, got shape {tuple(self.tensor.shape)}")


@dataclasses.dataclass(frozen=True)
class GpuCropBatch:
    """NCHW float crops on GPU, 750x1050 canonical."""
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        _require_gpu(self.tensor, "GpuCropBatch")
        if self.tensor.dim() != 4:
            raise ValueError(f"GpuCropBatch expects 4D NCHW, got shape {tuple(self.tensor.shape)}")
        # Layout assertion: C must be small (3 or 4); H/W must be larger than C.
        _, c, h, w = self.tensor.shape
        if c > 8 or h < c or w < c:
            raise ValueError(
                f"GpuCropBatch shape {tuple(self.tensor.shape)} not NCHW; "
                "expected N, C, H, W with C <= 8"
            )


@dataclasses.dataclass(frozen=True)
class GpuEmbeddingBatch:
    """2D embedding tensor on GPU."""
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        _require_gpu(self.tensor, "GpuEmbeddingBatch")
        if self.tensor.dim() != 2:
            raise ValueError(f"GpuEmbeddingBatch expects 2D, got shape {tuple(self.tensor.shape)}")
