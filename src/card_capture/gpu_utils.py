"""GPU acceleration utilities for PyTorch-based image processing."""

from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def get_device() -> torch.device:
    """
    Detect and return the best available device for PyTorch computation.
    
    Priority order:
    1. MPS (Metal Performance Shaders) - for M-series Apple Silicon Macs
    2. CUDA - for NVIDIA GPUs
    3. CPU - fallback
    
    Returns:
        torch.device: The selected device for PyTorch operations.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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
        device: torch device ("mps", "cuda", "cpu", or "auto") or torch.device object
    
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


def compute_histogram_stats_gpu(variance_values: list[float]) -> tuple[float, float]:
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


def is_histogram_outlier_gpu(variance: float, mean: float, std_dev: float, 
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
