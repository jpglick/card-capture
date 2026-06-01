"""GPU acceleration utilities for PyTorch-based image processing.

This module provides helpers for device selection (MPS/CPU) and hardware-accelerated
image metrics. CUDA is no longer supported; we target Apple Silicon exclusively.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F
import numpy as np
import cv2


import os

def _env_cpu_ok() -> bool:
    return os.environ.get("CC_ALLOW_CPU_FALLBACK", "0") == "1"


def get_device(allow_cpu_fallback: bool = False) -> torch.device:
    """Return the MPS device, or hard-fail if MPS is unavailable.

    v5.5 is Apple-Silicon-only. CPU is returned ONLY when the caller explicitly
    opts in via `allow_cpu_fallback` (mapped from PipelineConfig.allow_cpu_fallback
    / RuntimeMode 'cpu_debug').
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu_fallback:
        return torch.device("cpu")
    raise RuntimeError(
        "MPS GPU unavailable and allow_cpu_fallback is False. v5.5 is "
        "Apple-Silicon-only; set PipelineConfig.allow_cpu_fallback=True "
        "(or RuntimeMode 'cpu_debug') for CPU debug runs."
    )


def _frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a numpy BGR frame to a PyTorch tensor on the specified device.
    
    Converts BGR to grayscale and normalizes to [0, 1].
    
    Args:
        frame: numpy array of shape (H, W, 3) with values [0, 255]
        device: torch.device to place tensor on
        
    Returns:
        torch.Tensor of shape (1, 1, H, W) with values in [0, 1]
    """
    # Convert BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Normalize to [0, 1]
    gray_normalized = gray.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch and channel dimensions
    tensor = torch.from_numpy(gray_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    return tensor


def compute_variance_gpu(frame: np.ndarray, device: torch.device) -> float:
    """
    Compute Laplacian variance of a frame on GPU.
    
    This metric is commonly used to detect image blurriness. Higher variance
    indicates sharper, more detailed content.
    
    Args:
        frame: numpy BGR image of shape (H, W, 3) with values [0, 255]
        device: torch.device for computation
        
    Returns:
        float: Laplacian variance (always >= 0)
    """
    # Convert frame to tensor
    tensor = _frame_to_tensor(frame, device)
    
    # Create Laplacian kernel (3x3)
    laplacian_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # Apply Laplacian filter using convolution
    laplacian = F.conv2d(tensor, laplacian_kernel, padding=1)
    
    # Compute variance of the Laplacian response
    variance = torch.var(laplacian).item()
    
    # Scale to match cv2.Laplacian behavior (which operates on [0, 255] values)
    # Since we normalized to [0, 1], the response is scaled down by 255^2
    variance = variance * (255.0 ** 2)
    
    return float(variance)


def compute_sharpness_gpu(frame: np.ndarray, device: torch.device) -> float:
    """
    Compute sharpness metric of a frame on GPU.
    
    Uses Laplacian variance as the sharpness metric. Higher values indicate
    sharper images with more high-frequency content.
    
    Args:
        frame: numpy BGR image of shape (H, W, 3) with values [0, 255]
        device: torch.device for computation
        
    Returns:
        float: Sharpness metric (always >= 0)
    """
    # For now, reuse variance computation since they're the same metric
    return compute_variance_gpu(frame, device)


def compute_motion_gpu(frame1: np.ndarray, frame2: np.ndarray, device: Union[str, torch.device] = "auto") -> float:
    """Compute mean absolute pixel difference between consecutive frames (motion metric).
    
    Args:
        frame1: Previous frame (H, W) or (H, W, C), uint8 grayscale or color
        frame2: Current frame, same shape and type
        device: torch device ("mps", "cpu", or "auto") or torch.device object
    
    Returns:
        Mean pixel delta (0-255 scale for uint8 frames)
    
    Raises:
        ValueError: If frame shapes don't match
    """
    if device == "auto":
        device = get_device()
    
    # Validate frame shapes match
    if frame1.shape != frame2.shape:
        raise ValueError(f"frame1 and frame2 must have the same shape, got {frame1.shape} and {frame2.shape}")
    
    # Convert to grayscale if RGB
    if len(frame1.shape) == 3:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    if len(frame2.shape) == 3:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Convert to tensors on specified device
    t1 = torch.from_numpy(frame1).float().to(device)
    t2 = torch.from_numpy(frame2).float().to(device)
    
    # Mean absolute difference
    motion = torch.abs(t1 - t2).mean().item()
    
    return motion


def compute_histogram_stats(variance_values: list[float]) -> tuple[float, float]:
    """Compute mean and standard deviation of Laplacian variance distribution.
    
    Args:
        variance_values: List of Laplacian variance scores from video scan
    
    Returns:
        (mean, std_dev) of variance distribution
    """
    if not variance_values:
        return 0.0, 0.0
    
    values = np.array(variance_values, dtype=np.float32)
    return float(values.mean()), float(values.std())


def is_histogram_outlier(variance: float, mean: float, std_dev: float, 
                             sigma_threshold: float = 1.5) -> bool:
    """Check if a variance value is a statistical outlier.
    
    Frames with unusual variance (much higher or lower than typical) likely contain cards.
    
    Args:
        variance: Laplacian variance for current frame
        mean: Population mean from histogram stats
        std_dev: Population standard deviation
        sigma_threshold: Z-score threshold (default 1.5 = ±1.5σ band)
    
    Returns:
        True if |variance - mean| > sigma_threshold * std_dev
    """
    if std_dev == 0:
        return False  # No variation in data
    
    z_score = abs(variance - mean) / std_dev
    return z_score > sigma_threshold


def compute_edge_density_gpu(frame: np.ndarray, device: Union[str, torch.device] = "auto",
                             sobel_threshold: float = 50.0, 
                             edge_density_threshold: float = 0.15) -> tuple[float, bool]:
    """Compute edge density using Sobel operator for textured card detection.
    
    Args:
        frame: Input frame (H, W) grayscale or (H, W, C) color, uint8
        device: torch device ("mps", "cpu", or "auto")
        sobel_threshold: Edge magnitude threshold (0-255 scale), default 50
        edge_density_threshold: Fraction of high-edge pixels for detection, default 0.15
    
    Returns:
        (edge_density_fraction, is_high_edge) tuple
        - edge_density_fraction: Fraction of pixels with |Sobel| > threshold
        - is_high_edge: True if edge_density_fraction > edge_density_threshold
    """
    if device == "auto":
        device = get_device()
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Tensor conversion
    t = torch.from_numpy(frame).float().to(device)
    t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) for conv2d
    
    # Sobel X kernel
    sobel_x = torch.tensor([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    
    # Sobel Y kernel
    sobel_y = torch.tensor([
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0]
    ], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    
    # Apply Sobel kernels
    gx = F.conv2d(t, sobel_x, padding=1)
    gy = F.conv2d(t, sobel_y, padding=1)
    
    # Magnitude = sqrt(Gx^2 + Gy^2)
    magnitude = torch.sqrt(gx**2 + gy**2)
    
    # Scale magnitude to 0-255 range
    # Max Sobel response is sqrt(1020^2 + 1020^2) ≈ 1443
    magnitude = magnitude * (255.0 / 1443.0)
    
    # Count high-edge pixels
    high_edge_pixels = (magnitude > sobel_threshold).float().sum()
    total_pixels = float(magnitude.numel())
    
    edge_density = float(high_edge_pixels / total_pixels)
    is_high = edge_density > edge_density_threshold
    
    return edge_density, is_high


def estimate_batch_size(device: str = "auto", safety_margin: float = 0.4) -> int:
    """Estimate safe GPU batch size for sharpness scoring based on available VRAM.
    
    Queries GPU memory and estimates how many frames can be processed simultaneously
    for batched sharpness scoring, accounting for GPU memory overhead.
    
    Args:
        device: torch device ("mps", "cpu", or "auto")
        safety_margin: Fraction of VRAM to reserve (0.4 = 40% reserved, 60% usable)
    
    Returns:
        Safe batch size integer, clamped to [1, 128]
        - CPU: always 1
        - GPU: 32-128 depending on VRAM
    
    Raises:
        (caught internally) Any device query exceptions fallback to 32
    """
    try:
        # Convert device string to torch.device if needed
        if device == "auto":
            device_obj = get_device()
        else:
            device_obj = torch.device(device)
        
        # CPU always returns batch size 1
        if device_obj.type == "cpu":
            return 1
        
        # Query available VRAM based on device type
        if device_obj.type == "mps":
            # MPS (Mac) has no VRAM query API; estimate conservatively as 8GB
            available_vram_gb = 8.0
        else:
            # Unknown device type
            return 32
        
        # Apply safety margin: usable = available * (1 - safety_margin)
        usable_vram_gb = available_vram_gb * (1.0 - safety_margin)
        
        # Empirical per-frame cost: ~10 MB per frame for sharpness scoring
        per_frame_mb = 10.0
        per_frame_gb = per_frame_mb / 1024.0
        
        # Calculate batch size
        batch_size = int(usable_vram_gb / per_frame_gb)
        
        # Clamp to [1, 128]
        batch_size = max(1, min(128, batch_size))
        
        return batch_size
    
    except Exception:
        # Fallback on any error (missing device, etc.)
        return 32


def score_sharpness_batched(frames: list[np.ndarray], device: str = "auto", 
                            batch_size: int = 32) -> list[float]:
    """Score sharpness (Laplacian variance) for a batch of frames on GPU.
    
    Processes multiple frames in parallel on GPU for efficient batched computation.
    Uses Laplacian convolution variance as the sharpness metric.
    
    Args:
        frames: List of frames, each (H, W) grayscale or (H, W, C) color, uint8
        device: torch device ("mps", "cpu", or "auto")
        batch_size: Frames to process simultaneously, clamped to [1, 128]
    
    Returns:
        List of float variance scores in original frame order, same length as input frames.
        Empty list if input is empty.
    
    Raises:
        TypeError: If frames is not a list or contains non-ndarray items
    """
    # Handle empty input
    if not frames:
        return []
    
    # Convert device string to torch.device if needed
    if device == "auto":
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Clamp batch_size to valid range [1, 128]
    batch_size = max(1, min(128, batch_size))
    
    # Create Laplacian kernel (reused for all batches)
    laplacian_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    results = []
    
    # Process frames in batches
    for batch_idx in range(0, len(frames), batch_size):
        batch_frames = frames[batch_idx:batch_idx + batch_size]
        
        # Convert frames to grayscale tensors
        batch_tensors = []
        for frame in batch_frames:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Normalize to [0, 1] and convert to tensor
            gray_normalized = gray.astype(np.float32) / 255.0
            tensor = torch.from_numpy(gray_normalized).to(device)
            batch_tensors.append(tensor)
        
        # Stack into batch tensor (B, H, W)
        batch_tensor = torch.stack(batch_tensors)  # (B, H, W)
        # Add channel dimension: (B, 1, H, W)
        batch_tensor = batch_tensor.unsqueeze(1)
        
        # Apply Laplacian filter using convolution
        laplacian = F.conv2d(batch_tensor, laplacian_kernel, padding=1)
        
        # Compute variance per frame in batch
        # laplacian shape: (B, 1, H, W)
        batch_variances = []
        for i in range(laplacian.shape[0]):
            frame_laplacian = laplacian[i]  # (1, H, W)
            variance = torch.var(frame_laplacian).item()
            # Scale to match cv2.Laplacian behavior (255 scale)
            variance = variance * (255.0 ** 2)
            batch_variances.append(float(variance))
        
        results.extend(batch_variances)
    
    return results


def compute_presence_metrics_batched(
    frames: list[np.ndarray],
    device: str = "auto",
    batch_size: int = 32,
    sobel_threshold: float = 50.0,
    empty_pixel_threshold: float = 8.0,
) -> list[dict[str, float]]:
    """Compute low-resolution presence metrics for a batch of frames.

    The planner uses these metrics to derive per-video thresholds instead of
    relying on fixed global knobs.

    Returns one dict per frame with:
    - sharpness: Laplacian variance
    - variance: grayscale variance
    - edge_density: fraction of pixels above the Sobel threshold
    - empty_ratio: fraction of pixels close to black
    """
    if not frames:
        return []

    if device == "auto":
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    batch_size = max(1, min(128, batch_size))

    laplacian_kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).unsqueeze(0)

    results: list[dict[str, float]] = []

    for batch_idx in range(0, len(frames), batch_size):
        batch_frames = frames[batch_idx: batch_idx + batch_size]
        batch_tensors = []
        for frame in batch_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            gray_normalized = gray.astype(np.float32) / 255.0
            batch_tensors.append(torch.from_numpy(gray_normalized).to(device))

        batch_tensor = torch.stack(batch_tensors).unsqueeze(1)
        laplacian = F.conv2d(batch_tensor, laplacian_kernel, padding=1)
        gx = F.conv2d(batch_tensor, sobel_x, padding=1)
        gy = F.conv2d(batch_tensor, sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2) * (255.0 / 1443.0)

        sharpness = torch.var(laplacian, dim=(1, 2, 3)) * (255.0 ** 2)
        variance = torch.var(batch_tensor, dim=(1, 2, 3)) * (255.0 ** 2)
        edge_density = (magnitude > sobel_threshold).float().mean(dim=(1, 2, 3))
        empty_ratio = (batch_tensor <= (empty_pixel_threshold / 255.0)).float().mean(dim=(1, 2, 3))

        for i in range(batch_tensor.shape[0]):
            results.append(
                {
                    "sharpness": float(sharpness[i].item()),
                    "variance": float(variance[i].item()),
                    "edge_density": float(edge_density[i].item()),
                    "empty_ratio": float(empty_ratio[i].item()),
                }
            )

    return results


def require_device(requested: str) -> torch.device:
    """Resolve `requested` to a torch.device, raising rather than silently
    downgrading to CPU. `requested` of "cpu" is honored (explicit). "auto"
    requires the MPS GPU and raises if it is not present.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available "
                               "(this route is MPS-or-fail; no CPU fallback).")
        return torch.device("mps")
    # auto: require the MPS accelerator
    return resolve_required_gpu_device(requested)


def resolve_required_gpu_device(requested: str = "auto") -> torch.device:
    """Resolve to an MPS device, raising if no GPU is available.

    Apple Silicon only: CUDA is no longer supported.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("No MPS GPU available and this route forbids CPU fallback.")
