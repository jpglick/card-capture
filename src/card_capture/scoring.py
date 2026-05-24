from __future__ import annotations

import os
from typing import Optional, Sequence

import cv2
import numpy as np

from .models import QualityScore
from .occlusion_residual import compute_occlusion_residual_score
from . import gpu_utils

CARD_ASPECT_RATIO: float = 63.5 / 88.9  # ≈ 0.714 (width / height, standard trading card)
ASPECT_TOLERANCE: float = 0.15

# True when the operator explicitly permits silent CPU fallback.
_CPU_FALLBACK_ALLOWED = os.environ.get("CC_CUDA_ALLOW_CPU_FALLBACK", "0") == "1"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _laplacian_variance(gray: np.ndarray, device) -> float:
    """Compute Laplacian variance on CUDA GPU when available; cv2 otherwise.

    Only CUDA is used for the GPU path — MPS (Apple Silicon) uses float32
    which introduces tiny non-zero values on uniform images and produces
    numerically different results from cv2.Laplacian (float64 on uint8).
    MPS falls through to the cv2 path silently (it is still a GPU, just not
    the production target).  A warning is emitted only when the runtime has
    no CUDA and CC_CUDA_ALLOW_CPU_FALLBACK is not set.
    """
    if device.type == "cuda":
        # CUDA path: convert grayscale to BGR for gpu_utils (expects BGR input)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
        return gpu_utils.compute_variance_gpu(bgr, device)
    # Non-CUDA path (cpu or mps) — warn only on pure-CPU systems without opt-in.
    if device.type == "cpu" and not _CPU_FALLBACK_ALLOWED:
        import warnings
        warnings.warn(
            "QualityScorer: Laplacian sharpness running on CPU (no CUDA device found). "
            "Set CC_CUDA_ALLOW_CPU_FALLBACK=1 to suppress this warning.",
            RuntimeWarning,
            stacklevel=3,
        )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class QualityScorer:
    def __init__(self, target_pixels: int = 600 * 900, device: str = "auto"):
        self.target_pixels = target_pixels
        if device == "auto":
            self._device = gpu_utils.get_device()
        else:
            import torch
            self._device = torch.device(device)

    def score(
        self,
        image: np.ndarray,
        detection_confidence: float,
        prior_frames: Optional[Sequence[np.ndarray]] = None,
        novelty: float = 1.0,
    ) -> QualityScore:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        laplacian_variance = _laplacian_variance(gray, self._device)
        sharpness = _clamp(laplacian_variance / 1000.0)

        overexposed_fraction = float((gray >= 245).mean())
        glare = _clamp(1.0 - overexposed_fraction * 4.0)

        h, w = image.shape[:2]
        actual_ratio = (w / h) if h > 0 else 0.0
        aspect_ratio = _clamp(
            1.0 - abs(actual_ratio - CARD_ASPECT_RATIO) / ASPECT_TOLERANCE
        )

        size = _clamp((h * w) / float(self.target_pixels))

        # Grayscale std-dev rewards textured fronts over plain backs
        complexity = _clamp(float(gray.std()) / 80.0)

        # Border purity: the outer ~3% ring should be relatively uniform (real
        # cards have a clean border). A noisy border ring signals a hand/finger
        # intruding into the rectified crop. We compare the border ring's
        # std-dev to the interior std-dev; a clean border has ring_std much
        # lower than interior_std, an occluded one does not.
        border_purity = _border_purity_score(gray)

        # Spatial glare: largest saturated blob via connected-component analysis.
        # Distinguishes scattered specular reflections from large blowout regions.
        spatial_glare = _spatial_glare_score(image)

        # Occlusion residual: detect interior occlusions via per-tile median residuals.
        # Requires prior frames; if unavailable, score is neutral (1.0).
        occlusion = 1.0
        if prior_frames:
            occlusion = compute_occlusion_residual_score(gray, prior_frames)

        confidence = _clamp(float(detection_confidence))

        total = (
            sharpness * 0.35
            + aspect_ratio * 0.20
            + novelty * 0.15
            + glare * 0.10
            + border_purity * 0.10
            + size * 0.05
            + spatial_glare * 0.05
            + complexity * 0.00
            + occlusion * 0.00
        )
        components = {
            "sharpness": round(sharpness, 6),
            "glare": round(glare, 6),
            "aspect_ratio": round(aspect_ratio, 6),
            "size": round(size, 6),
            "complexity": round(complexity, 6),
            "border_purity": round(border_purity, 6),
            "spatial_glare": round(spatial_glare, 6),
            "occlusion": round(occlusion, 6),
            "confidence": round(confidence, 6),
            "novelty": round(novelty, 6),
        }
        return QualityScore(total=round(total, 6), components=components)


def _spatial_glare_score(image: np.ndarray) -> float:
    """Return [0, 1] where 1 = no glare, 0 = severe glare (large saturated blob).

    Strategy: use connected-component analysis on saturated pixels (V > 240 in HSV)
    to find the largest contiguous blob. Normalize by frame area and clip to [0, 1].

    Formula:
    - blob_fraction = largest_blob_area / frame_area
    - score = clip(1.0 - blob_fraction × 10, 0, 1)

    Examples:
    - blob_fraction=0 → score=1.0 (no glare)
    - blob_fraction=0.1 → score~0.0 (severe glare: 10% of frame is saturated blob)
    """
    h, w = image.shape[:2]
    frame_area = h * w

    # Convert to HSV and extract V channel
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
    else:
        v_channel = image

    # Create binary mask of saturated pixels (V > 240)
    saturated_mask = (v_channel > 240).astype(np.uint8)

    # Use connected components with stats to find all blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        saturated_mask, connectivity=8
    )

    if num_labels <= 1:
        # No saturated pixels (label 0 is background)
        return 1.0

    # stats[:, cv2.CC_STAT_AREA] gives area of each component
    # Ignore label 0 (background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_blob_area = float(areas.max())

    # Normalize: blob_fraction = largest_area / frame_area
    blob_fraction = largest_blob_area / frame_area

    # Score: clip(1.0 - blob_fraction × 10, 0, 1)
    # - blob_fraction ≤ 0.1 → score ≈ 0.0
    # - blob_fraction=0 → score=1.0
    score = _clamp(1.0 - blob_fraction * 10.0, low=0.0, high=1.0)
    return score


def _border_purity_score(gray: np.ndarray) -> float:
    """Return [0, 1] where 1 = clean uniform border, 0 = noisy / occluded border.

    Strategy: compare std-dev of the outer ~3% ring to std-dev of the inner
    region. On a real card the border is a near-uniform white edge so ring
    std-dev is small; if a finger or other texture has intruded, ring std-dev
    spikes. Returns `1 - clamp(ring_std / max(interior_std, eps))`."""
    h, w = gray.shape[:2]
    bw = max(2, int(round(min(h, w) * 0.03)))
    if h <= 2 * bw or w <= 2 * bw:
        return 0.5  # crop too small to evaluate; neutral
    ring_mask = np.ones((h, w), dtype=bool)
    ring_mask[bw:h - bw, bw:w - bw] = False
    ring_vals = gray[ring_mask]
    interior_vals = gray[bw:h - bw, bw:w - bw].ravel()
    if ring_vals.size == 0 or interior_vals.size == 0:
        return 0.5
    ring_std = float(ring_vals.std())
    interior_std = float(interior_vals.std()) + 1e-3
    # Clean cards: ring_std ≪ interior_std. Occluded: ring_std comparable or larger.
    ratio = ring_std / interior_std
    return _clamp(1.0 - min(1.0, ratio))
