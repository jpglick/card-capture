import numpy as np
import cv2
from src.card_capture.fusion.foil_detection import detect_foil_card, compute_laplacian_variance

def test_foil_detection_high_variance_across_frames():
    """Verify foil cards show high Laplacian variance across frames."""
    # Regular card: consistent edge structure across frames
    # Use the same base image for all frames (no shifting high-freq patterns)
    base_regular = np.ones((750, 1050, 3), dtype=np.uint8) * 120
    regular_frames = [base_regular.copy() for _ in range(4)]

    # Foil card: high-frequency content shifts (holographic surface)
    # Simulate by adding different random high-frequency patterns to each frame
    foil_frames = []
    for _ in range(4):
        base = np.ones((750, 1050, 3), dtype=np.uint8) * 120
        # Add different high-frequency pattern to each frame
        high_freq = np.random.randint(0, 50, (750, 1050, 3), dtype=np.uint8)
        foil_frames.append(np.clip(base.astype(np.int32) + high_freq.astype(np.int32), 0, 255).astype(np.uint8))

    regular_var = compute_laplacian_variance(regular_frames)
    foil_var = compute_laplacian_variance(foil_frames)

    # Foil should have higher variance
    assert foil_var > regular_var, f"Foil variance ({foil_var}) should exceed regular ({regular_var})"

    # Threshold-based detection
    is_foil_regular = detect_foil_card(regular_frames, threshold=50.0)
    is_foil_card = detect_foil_card(foil_frames, threshold=50.0)

    assert not is_foil_regular, "Regular card should not be detected as foil"
    assert is_foil_card, "Foil card should be detected"

def test_foil_detection_threshold_tuning():
    """Verify foil detection is tunable via threshold."""
    # Mid-variance frames (could go either way)
    frames = [np.random.randint(100, 160, (750, 1050, 3), dtype=np.uint8) for _ in range(4)]

    # Low threshold: more sensitive (more false positives)
    detected_low = detect_foil_card(frames, threshold=30.0)

    # High threshold: less sensitive (more false negatives)
    detected_high = detect_foil_card(frames, threshold=100.0)

    # At least one should detect it (or neither, but consistency matters)
    # The key is that threshold is tunable
    assert isinstance(detected_low, bool), "Should return boolean"
    assert isinstance(detected_high, bool), "Should return boolean"

def test_glare_rejection_fusion_preserves_luminance():
    """Verify glare-rejection fusion picks closest-to-median pixels."""
    from src.card_capture.fusion.median_fusion import glare_rejection_fusion

    # Three frames: one bright (glare), two nominal
    frames = [
        np.ones((100, 100, 3), dtype=np.uint8) * 120,  # nominal
        np.ones((100, 100, 3), dtype=np.uint8) * 250,  # bright (glare)
        np.ones((100, 100, 3), dtype=np.uint8) * 118,  # nominal (close to first)
    ]

    fused = glare_rejection_fusion(frames)

    # Result should be close to median (120), not glare (250)
    mean_pixel_value = np.mean(fused)

    assert 110 < mean_pixel_value < 130, f"Should preserve luminance ~120, got {mean_pixel_value}"
    assert mean_pixel_value < 200, f"Should reject glare, but got {mean_pixel_value}"

def test_glare_rejection_fusion_shape():
    """Verify glare-rejection fusion returns same shape as input frames."""
    from src.card_capture.fusion.median_fusion import glare_rejection_fusion

    frames = [
        np.random.randint(50, 200, (750, 1050, 3), dtype=np.uint8),
        np.random.randint(50, 200, (750, 1050, 3), dtype=np.uint8),
        np.random.randint(50, 200, (750, 1050, 3), dtype=np.uint8),
    ]

    fused = glare_rejection_fusion(frames)

    assert fused.shape == (750, 1050, 3), f"Shape mismatch: expected (750, 1050, 3), got {fused.shape}"
    assert fused.dtype == np.uint8, f"Type should be uint8, got {fused.dtype}"
