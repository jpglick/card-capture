from __future__ import annotations

import numpy as np
import pytest
import torch

from card_capture.core.gpu_utils import (
    get_device, 
    compute_variance_gpu, 
    compute_sharpness_gpu, 
    compute_motion_gpu,
    compute_histogram_stats,
    is_histogram_outlier,
    compute_edge_density_gpu,
    estimate_batch_size,
    score_sharpness_batched,
    compute_presence_metrics_batched,
)


# ---------------------------------------------------------------------------
# Device Detection Tests
# ---------------------------------------------------------------------------

def test_get_device_returns_valid_device():
    """Verify that get_device() returns a valid torch.device."""
    device = get_device(allow_cpu_fallback=True)
    assert isinstance(device, torch.device)
    # Device should be one of MPS or CPU
    assert device.type in ("mps", "cpu")


# ---------------------------------------------------------------------------
# Variance Computation Tests
# ---------------------------------------------------------------------------

def test_compute_variance_gpu_returns_scalar():
    """Verify that compute_variance_gpu returns a float >= 0."""
    device = get_device(allow_cpu_fallback=True)
    
    # Create a test frame (8-bit BGR image, typical from OpenCV)
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    variance = compute_variance_gpu(frame, device)
    
    # Should return a float scalar
    assert isinstance(variance, float)
    # Variance should be non-negative
    assert variance >= 0


def test_compute_variance_gpu_high_variance():
    """Verify variance is higher for high-frequency content."""
    device = get_device(allow_cpu_fallback=True)
    
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
    device = get_device(allow_cpu_fallback=True)
    
    # Create a test frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    sharpness = compute_sharpness_gpu(frame, device)
    
    # Should return a float scalar
    assert isinstance(sharpness, float)
    # Sharpness should be non-negative
    assert sharpness >= 0


def test_compute_sharpness_gpu_blurry_vs_sharp():
    """Verify sharpness is lower for blurry images and higher for sharp."""
    device = get_device(allow_cpu_fallback=True)
    
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
    mean, std = compute_histogram_stats(values)
    assert abs(mean - 100.0) < 0.01
    assert std < 0.01


def test_histogram_stats_normal_distribution():
    """Known distribution should compute correct stats."""
    values = [100.0, 105.0, 110.0, 115.0, 120.0]  # mean=110, known std
    mean, std = compute_histogram_stats(values)
    assert abs(mean - 110.0) < 0.1
    assert std > 0  # Should have variation


def test_is_histogram_outlier_within_band():
    """Value within ±σ band should not trigger outlier."""
    is_outlier = is_histogram_outlier(variance=105.0, mean=100.0, 
                                            std_dev=10.0, sigma_threshold=1.5)
    # z_score = |105-100|/10 = 0.5 < 1.5
    assert not is_outlier


def test_is_histogram_outlier_outside_band():
    """Value outside ±σ band should trigger outlier."""
    is_outlier = is_histogram_outlier(variance=120.0, mean=100.0, 
                                            std_dev=10.0, sigma_threshold=1.5)
    # z_score = |120-100|/10 = 2.0 > 1.5
    assert is_outlier


def test_is_histogram_outlier_zero_std():
    """Zero std dev (no variation) should return False."""
    is_outlier = is_histogram_outlier(variance=100.0, mean=100.0, 
                                            std_dev=0.0, sigma_threshold=1.5)
    assert not is_outlier


# ---------------------------------------------------------------------------
# Edge Density Detection Tests
# ---------------------------------------------------------------------------

def test_edge_density_blank_frame():
    """Uniform frame should have near-zero edge density."""
    frame = np.full((50, 50), 128, dtype=np.uint8)
    density, is_high = compute_edge_density_gpu(frame, sobel_threshold=50.0, 
                                                edge_density_threshold=0.15)
    assert density < 0.10
    assert not is_high


def test_edge_density_checkerboard():
    """Checkerboard pattern should have detectable high edge density."""
    frame = np.zeros((100, 100), dtype=np.uint8)
    frame[::2, ::2] = 255  # Checkerboard
    density, is_high = compute_edge_density_gpu(frame, sobel_threshold=50.0,
                                                edge_density_threshold=0.01)
    assert density > 0.01  # Checkerboard has ~2% edge pixels at grid lines
    assert is_high


def test_edge_density_threshold_varies_detection():
    """Higher threshold should reduce detections."""
    frame = np.zeros((100, 100), dtype=np.uint8)
    frame[::2, ::2] = 200
    
    # Loose threshold
    _, is_high_loose = compute_edge_density_gpu(frame, sobel_threshold=30.0,
                                                 edge_density_threshold=0.005)
    # Strict threshold
    _, is_high_strict = compute_edge_density_gpu(frame, sobel_threshold=150.0,
                                                  edge_density_threshold=0.40)
    
    assert is_high_loose  # Loose should detect
    assert not is_high_strict  # Strict should not


def test_presence_metrics_batched_returns_per_frame_metrics():
    frames = [
        np.full((64, 64, 3), 16, dtype=np.uint8),
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.full((64, 64, 3), 220, dtype=np.uint8),
    ]
    metrics = compute_presence_metrics_batched(frames, device="cpu", batch_size=2)

    assert len(metrics) == 3
    assert all(set(item.keys()) == {"sharpness", "variance", "edge_density", "empty_ratio"} for item in metrics)
    assert metrics[1]["empty_ratio"] >= metrics[0]["empty_ratio"]
    assert metrics[2]["variance"] >= metrics[1]["variance"]


def test_edge_density_rgb_frame():
    """RGB input should auto-convert to grayscale."""
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[::2, ::2] = [255, 255, 255]  # White checkerboard
    density, is_high = compute_edge_density_gpu(frame, sobel_threshold=50.0,
                                                edge_density_threshold=0.01)
    assert is_high


# ---------------------------------------------------------------------------
# Batch Size Estimation Tests
# ---------------------------------------------------------------------------

def test_estimate_batch_size_cpu_returns_one():
    """CPU device should always return 1."""
    batch_size = estimate_batch_size(device="cpu")
    assert batch_size == 1


def test_estimate_batch_size_clamped_to_max():
    """Batch size should never exceed 128."""
    batch_size = estimate_batch_size(device="auto")
    assert batch_size <= 128


def test_estimate_batch_size_clamped_to_min():
    """Batch size should never be below 1."""
    batch_size = estimate_batch_size(device="auto")
    assert batch_size >= 1


def test_estimate_batch_size_type_validation():
    """Return value should be integer."""
    batch_size = estimate_batch_size(device="cpu")
    assert isinstance(batch_size, int)


# ---------------------------------------------------------------------------
# Batched Sharpness Scoring Tests
# ---------------------------------------------------------------------------

def test_score_sharpness_batched_empty():
    """Empty frame list should return empty results."""
    results = score_sharpness_batched([])
    assert results == []


def test_score_sharpness_batched_single():
    """Single frame should work (batch_size=1)."""
    frame = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    results = score_sharpness_batched([frame], batch_size=1)
    assert len(results) == 1
    assert isinstance(results[0], float)
    assert results[0] >= 0


def test_score_sharpness_batched_batch_size():
    """Large batch should produce correct count."""
    frames = [np.random.randint(0, 256, (50, 50), dtype=np.uint8) for _ in range(10)]
    results = score_sharpness_batched(frames, batch_size=4)
    assert len(results) == 10
    # All should be floats >= 0
    for score in results:
        assert isinstance(score, float)
        assert score >= 0


def test_score_sharpness_batched_matches_sequential():
    """Batched results should match sequential computation (within float tolerance)."""
    frames = [np.random.randint(50, 200, (50, 50), dtype=np.uint8) for _ in range(3)]
    
    # Batched
    batched_results = score_sharpness_batched(frames, batch_size=10)
    
    # Sequential (batch_size=1)
    sequential_results = score_sharpness_batched(frames, batch_size=1)
    
    # Should match within floating point tolerance
    assert len(batched_results) == len(sequential_results)
    for batched, sequential in zip(batched_results, sequential_results):
        assert abs(batched - sequential) < 1e-4
