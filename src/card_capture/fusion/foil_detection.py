import os
import warnings
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Calibrated via scripts/calibrate_foil_threshold.py
# Threshold for detecting foil/holographic cards based on Laplacian variance across frames.
# Calibration on synthetic fixture sets (3 foil + 3 non-foil groups) showed perfect
# separation across the range 10.0-80.0, selecting 50.0 as a robust midpoint.
DEFAULT_FOIL_THRESHOLD = 50.0

# Resolve GPU device once at module load time (MPS > CPU).
from card_capture import gpu_utils as _gpu_utils
_device = _gpu_utils.get_device(allow_cpu_fallback=_gpu_utils._env_cpu_ok())


def _compute_laplacian_variance_gpu(frames: list[np.ndarray]) -> float:
    """Compute mean cross-frame Laplacian variance on GPU.

    Mirrors the CPU path: computes |Laplacian| per frame, stacks them,
    computes pixel-wise variance across frames, and returns the mean variance.
    Scaled by 255^2 so results are numerically equivalent to the cv2 path
    (which operates on uint8 [0,255] values instead of normalised [0,1]).
    """
    laplacian_kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        dtype=torch.float32,
        device=_device,
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

    lap_mags = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        # Normalise to [0, 1]; Laplacian response is therefore in [0, 1] scale.
        t = torch.from_numpy(gray.astype(np.float32) / 255.0).to(_device)
        t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        lap = F.conv2d(t, laplacian_kernel, padding=1)
        lap_mags.append(lap.abs().squeeze())  # (H, W)

    stacked = torch.stack(lap_mags, dim=0)  # (N, H, W)
    variance_map = torch.var(stacked, dim=0)
    # Scale to match cv2 magnitudes: cv2 works on [0,255] uint8, so its variance
    # is 255^2 larger than the normalised [0,1] version.
    return float(variance_map.mean().item()) * (255.0 ** 2)


def _compute_laplacian_variance_cpu(frames: list[np.ndarray]) -> float:
    """CPU fallback: mean cross-frame Laplacian variance via cv2."""
    laplacian_mags = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_mags.append(np.abs(laplacian))
    stacked = np.stack(laplacian_mags, axis=0)
    variance_map = np.var(stacked, axis=0)
    return float(np.mean(variance_map))


def compute_laplacian_variance(frames: list[np.ndarray]) -> float:
    """
    Compute the mean variance of Laplacian magnitudes across frames via cv2.

    High-frequency content that shifts between frames indicates a foil/holographic surface.
    The former CUDA GPU path was dead on Apple Silicon (MPS uses float32 and
    fell through to cv2 anyway); v5.5 is Apple-Silicon-only, so it is removed.

    Args:
        frames: List of BGR frames (typically 4 canonical frames)

    Returns:
        Mean spatial variance of Laplacian magnitudes across all frames
    """
    if len(frames) < 2:
        return 0.0

    return _compute_laplacian_variance_cpu(frames)


def detect_foil_card(frames: list[np.ndarray], threshold: float = DEFAULT_FOIL_THRESHOLD) -> bool:
    """
    Detect if a card is foil/holographic based on Laplacian variance.

    Foil cards have high-frequency content that shifts between frames,
    resulting in high variance of Laplacian magnitudes.

    Args:
        frames: List of BGR frames
        threshold: Variance threshold above which card is considered foil (default 50.0)

    Returns:
        True if card is foil, False otherwise
    """
    if len(frames) < 2:
        return False
    variance = compute_laplacian_variance(frames)
    return variance > threshold
