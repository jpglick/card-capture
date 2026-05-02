from __future__ import annotations

import numpy as np
import pytest
import torch

from card_capture.gpu_utils import get_device, compute_variance_gpu, compute_sharpness_gpu


# ---------------------------------------------------------------------------
# Device Detection Tests
# ---------------------------------------------------------------------------

def test_get_device_returns_valid_device():
    """Verify that get_device() returns a valid torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)
    # Device should be one of MPS, CUDA, or CPU
    assert device.type in ("mps", "cuda", "cpu")


# ---------------------------------------------------------------------------
# Variance Computation Tests
# ---------------------------------------------------------------------------

def test_compute_variance_gpu_returns_scalar():
    """Verify that compute_variance_gpu returns a float >= 0."""
    device = get_device()
    
    # Create a test frame (8-bit BGR image, typical from OpenCV)
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    variance = compute_variance_gpu(frame, device)
    
    # Should return a float scalar
    assert isinstance(variance, float)
    # Variance should be non-negative
    assert variance >= 0


def test_compute_variance_gpu_high_variance():
    """Verify variance is higher for high-frequency content."""
    device = get_device()
    
    # Low variance: smooth gradient
    low_var_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    low_var = compute_variance_gpu(low_var_frame, device)
    
    # High variance: checkerboard pattern
    high_var_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    high_var_frame[::2, ::2] = 255
    high_var = compute_variance_gpu(high_var_frame, device)
    
    # High variance frame should have higher variance
    assert high_var > low_var


# ---------------------------------------------------------------------------
# Sharpness Computation Tests
# ---------------------------------------------------------------------------

def test_compute_sharpness_gpu_returns_scalar():
    """Verify that compute_sharpness_gpu returns a float >= 0."""
    device = get_device()
    
    # Create a test frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    sharpness = compute_sharpness_gpu(frame, device)
    
    # Should return a float scalar
    assert isinstance(sharpness, float)
    # Sharpness should be non-negative
    assert sharpness >= 0


def test_compute_sharpness_gpu_blurry_vs_sharp():
    """Verify sharpness is lower for blurry images and higher for sharp."""
    device = get_device()
    
    # Sharp image: high-frequency content
    sharp_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sharp_frame[::2, ::2] = 255
    sharp_val = compute_sharpness_gpu(sharp_frame, device)
    
    # Blurry image: smooth content
    blurry_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    blurry_val = compute_sharpness_gpu(blurry_frame, device)
    
    # Sharp image should have higher sharpness
    assert sharp_val > blurry_val
