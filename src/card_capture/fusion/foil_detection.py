import numpy as np
import cv2

# Calibrated via scripts/calibrate_foil_threshold.py
# Threshold for detecting foil/holographic cards based on Laplacian variance across frames.
# Calibration on synthetic fixture sets (3 foil + 3 non-foil groups) showed perfect
# separation across the range 10.0-80.0, selecting 50.0 as a robust midpoint.
DEFAULT_FOIL_THRESHOLD = 50.0


def compute_laplacian_variance(frames: list[np.ndarray]) -> float:
    """
    Compute the mean variance of Laplacian magnitudes across frames.

    High-frequency content that shifts between frames indicates a foil/holographic surface.

    Args:
        frames: List of BGR frames (typically 4 canonical frames)

    Returns:
        Mean spatial variance of Laplacian magnitudes across all frames
    """
    if len(frames) < 2:
        return 0.0

    # Convert frames to grayscale and compute Laplacian for each
    laplacian_mags = []
    for frame in frames:
        # Convert BGR to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Laplacian high-pass filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Take absolute value (magnitude)
        mag = np.abs(laplacian)

        laplacian_mags.append(mag)

    # Stack magnitudes across frames (frames × height × width)
    stacked = np.stack(laplacian_mags, axis=0)

    # Compute spatial variance per pixel (across time dimension)
    # This gives us a (height × width) variance map
    variance_map = np.var(stacked, axis=0)

    # Return mean variance across all pixels
    mean_variance = np.mean(variance_map)

    return float(mean_variance)


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
