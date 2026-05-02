from __future__ import annotations

import numpy as np
import pytest
import torch

from card_capture.gpu_utils import (
    get_device, 
    compute_variance_gpu, 
    compute_sharpness_gpu, 
    compute_motion_gpu,
    compute_histogram_stats_gpu,
    is_histogram_outlier_gpu
)


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


# ---------------------------------------------------------------------------
# Motion Detection Tests
# ---------------------------------------------------------------------------

def test_motion_detection_no_motion():
    """Identical frames should have motion ~0."""
    frame = np.full((50, 50), 128, dtype=np.uint8)
    motion = compute_motion_gpu(frame, frame)
    assert motion < 0.1


def test_motion_detection_high_motion():
    """Maximum difference should show high motion."""
    frame1 = np.zeros((50, 50), dtype=np.uint8)
    frame2 = np.full((50, 50), 255, dtype=np.uint8)
    motion = compute_motion_gpu(frame1, frame2)
    assert motion > 250


def test_motion_detection_rgb_input():
    """RGB input should auto-convert to grayscale."""
    frame1 = np.zeros((50, 50, 3), dtype=np.uint8)
    frame2 = np.full((50, 50, 3), 200, dtype=np.uint8)
    motion = compute_motion_gpu(frame1, frame2)
    assert 190 < motion < 210  # ~200 ± noise


def test_motion_detection_shape_mismatch():
    """Shape mismatch should raise ValueError."""
    frame1 = np.zeros((50, 50), dtype=np.uint8)
    frame2 = np.zeros((60, 60), dtype=np.uint8)
    with pytest.raises(ValueError, match="must have the same shape"):
        compute_motion_gpu(frame1, frame2)


# ---------------------------------------------------------------------------
# Histogram Outlier Detection Tests
# ---------------------------------------------------------------------------

def test_histogram_stats_uniform():
    """Uniform variance values should have zero std dev."""
    values = [100.0] * 10
    mean, std = compute_histogram_stats_gpu(values)
    assert abs(mean - 100.0) < 0.01
    assert std < 0.01


def test_histogram_stats_normal_distribution():
    """Known distribution should compute correct stats."""
    values = [100.0, 105.0, 110.0, 115.0, 120.0]  # mean=110, known std
    mean, std = compute_histogram_stats_gpu(values)
    assert abs(mean - 110.0) < 0.1
    assert std > 0  # Should have variation


def test_is_histogram_outlier_within_band():
    """Value within ±σ band should not trigger outlier."""
    is_outlier = is_histogram_outlier_gpu(variance=105.0, mean=100.0, 
                                           std_dev=10.0, sigma_threshold=1.5)
    # z_score = |105-100|/10 = 0.5 < 1.5
    assert not is_outlier


def test_is_histogram_outlier_outside_band():
    """Value outside ±σ band should trigger outlier."""
    is_outlier = is_histogram_outlier_gpu(variance=120.0, mean=100.0, 
                                           std_dev=10.0, sigma_threshold=1.5)
    # z_score = |120-100|/10 = 2.0 > 1.5
    assert is_outlier


def test_is_histogram_outlier_zero_std():
    """Zero std dev (no variation) should return False."""
    is_outlier = is_histogram_outlier_gpu(variance=100.0, mean=100.0, 
                                           std_dev=0.0, sigma_threshold=1.5)
    assert not is_outlier
