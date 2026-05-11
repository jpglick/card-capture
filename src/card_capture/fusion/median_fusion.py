"""
Fusion strategies for combining multiple frames of a card capture.
"""
import numpy as np


def glare_rejection_fusion(frames: list[np.ndarray]) -> np.ndarray:
    """
    Glare-rejection fusion for foil/holographic cards.

    For each pixel, compute the per-pixel median across all frames, then pick
    the frame whose pixel value is closest to the median. This preserves the
    card's natural luminance while rejecting bright outliers (glare).

    Algorithm:
    1. Compute per-pixel median across all frames
    2. For each pixel, measure distance to median in each frame
    3. Pick the frame whose pixel is closest to the median
    4. Use that frame's pixel value in the output

    Args:
        frames: List of BGR frames (typically 4 canonical frames), each uint8

    Returns:
        Fused uint8 BGR frame where each pixel comes from the frame closest
        to the per-pixel median, rejecting glare outliers.
    """
    # Stack frames: shape (num_frames, height, width, 3)
    stacked = np.stack(frames, axis=0).astype(np.float32)

    # Compute per-pixel median across frames (axis 0)
    # Result shape: (height, width, 3)
    median = np.median(stacked, axis=0)

    # For each pixel, compute distance to median in each frame
    # distances shape: (num_frames, height, width, 3)
    distances = np.abs(stacked - median)

    # Sum distance across channels to get scalar distance per pixel per frame
    # Result shape: (num_frames, height, width)
    distance_sum = np.sum(distances, axis=3)

    # Find the frame with minimum distance for each pixel
    # Result shape: (height, width)
    closest_frame_idx = np.argmin(distance_sum, axis=0)

    # Build output by picking pixels from the closest frame
    # For each (h, w) position, use the pixel from frames[closest_frame_idx[h, w]]
    height, width = closest_frame_idx.shape
    output = np.zeros((height, width, 3), dtype=np.float32)

    for h in range(height):
        for w in range(width):
            frame_idx = closest_frame_idx[h, w]
            output[h, w] = stacked[frame_idx, h, w]

    # Convert back to uint8
    return np.clip(output, 0, 255).astype(np.uint8)
