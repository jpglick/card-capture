"""GPU acceleration utilities for PyTorch-based image processing."""

from __future__ import annotations

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
