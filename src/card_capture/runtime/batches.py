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
    if t.device.type != "mps":
        raise WrongDeviceError(f"{name} requires an MPS tensor; got device={t.device}")


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


# --- Approved CPU-export boundary helpers ---------------------------------
#
# Each helper is the SOLE legal `.cpu()` / `.numpy()` / cv2.imwrite call site
# inside files tagged GPU-resident. The GPU-strict AST scanner (Phase 2) will
# fail if a tagged file calls `tensor.cpu()` outside these helpers.

import cv2  # noqa: E402  (intentional: this module owns the boundary)
try:
    import imagehash  # type: ignore[import-not-found]
except ImportError:
    imagehash = None
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pathlib import Path  # noqa: E402


def _to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """Internal: NCHW float [0,1] GPU -> HWC uint8 CPU."""
    x = t.detach()
    x = (x.clamp(0, 1) * 255).to(torch.uint8)
    x = x.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
    return x.cpu().numpy()  # boundary


def to_cpu_for_score(crops: GpuCropBatch) -> list[dict[str, float]]:
    """Reduce a crop batch to per-crop scalar quality scores.

    Stage 7 boundary. Returns one dict per crop matching QualityScore fields.
    """
    arr = _to_uint8_hwc(crops.tensor)  # (N, H, W, C)
    out: list[dict[str, float]] = []
    for img in arr:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        glare = float((gray > 240).mean())
        h, w = gray.shape
        aspect = float(h / w) if w else 0.0
        size = float(h * w)
        complexity = float(cv2.Canny(gray, 100, 200).mean())
        # border purity: variance of outer 8px ring
        ring = np.concatenate([gray[:8, :].ravel(), gray[-8:, :].ravel(),
                                gray[:, :8].ravel(), gray[:, -8:].ravel()])
        border_purity = float(1.0 / (ring.std() + 1.0))
        out.append({
            "sharpness": sharpness,
            "glare": glare,
            "aspect_ratio": aspect,
            "size": size,
            "complexity": complexity,
            "border_purity": border_purity,
            "confidence": 1.0,        # filled by detector
        })
    return out


def to_cpu_for_phash(crops: GpuCropBatch) -> list[str]:
    """Stage 8/10 boundary: perceptual hashes as hex strings."""
    arr = _to_uint8_hwc(crops.tensor)
    return [str(imagehash.phash(Image.fromarray(img))) for img in arr]


def to_cpu_for_fuse(crops: GpuCropBatch) -> np.ndarray:
    """Stage 9 boundary: NHWC uint8 array for fusion candidate selection."""
    return _to_uint8_hwc(crops.tensor)


def to_cpu_for_dedup(embeddings: GpuEmbeddingBatch) -> np.ndarray:
    """Stage 10 boundary: float32 numpy embeddings for cosine comparison."""
    return embeddings.tensor.detach().to(torch.float32).cpu().numpy()


def to_cpu_for_export(
    crops: GpuCropBatch, out_dir: Path, basenames: list[str]
) -> list[Path]:
    """Final export boundary: writes 750x1050 PNGs to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = _to_uint8_hwc(crops.tensor)
    if len(arr) != len(basenames):
        raise ValueError(f"basenames length {len(basenames)} != batch size {len(arr)}")
    paths: list[Path] = []
    for img, base in zip(arr, basenames):
        # Image is RGB from _to_uint8_hwc
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        p = out_dir / f"{base}.png"
        cv2.imwrite(str(p), bgr)
        paths.append(p)
    return paths
