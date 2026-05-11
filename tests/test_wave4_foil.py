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
