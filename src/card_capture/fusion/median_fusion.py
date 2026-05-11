"""
Fusion strategies for combining multiple frames of a card capture.
"""
import numpy as np
import cv2


def glare_rejection_fusion(frames: list[np.ndarray]) -> np.ndarray:
    """
    Glare-rejection fusion for foil/holographic cards.

    For each pixel, compute the per-pixel median across all frames, then pick
    the frame whose pixel value is closest to the median LUMINANCE. This
    preserves the card's natural luminance while rejecting bright outliers
    (glare). Selection is based on the L (luminance) channel of Lab color space,
    since glare is primarily a luminance phenomenon; a color-shifted frame with
    correct luminance should not lose to a luminance-glaring frame.

    Algorithm:
    1. Convert all frames to Lab color space
    2. Compute per-pixel median luminance (L channel) across all frames
    3. For each pixel, measure distance to median luminance in each frame
    4. Pick the frame with minimum L distance for each pixel
    5. Use that frame's BGR pixel value in the output (not the Lab pixel)

    Args:
        frames: List of BGR frames (typically 4 canonical frames), each uint8

    Returns:
        Fused uint8 BGR frame where each pixel comes from the frame with
        closest luminance to the per-pixel median, rejecting glare outliers.
    """
    # Stack frames: shape (num_frames, height, width, 3)
    stacked_bgr = np.stack(frames, axis=0).astype(np.float32)

    # Convert each frame to Lab color space for distance computation
    # Lab is perceptually uniform; glare is primarily a luminance (L) phenomenon
    stacked_lab = np.zeros_like(stacked_bgr)
    for i in range(stacked_bgr.shape[0]):
        # cv2.cvtColor expects uint8 input; convert back temporarily
        bgr_uint8 = np.clip(stacked_bgr[i], 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2Lab).astype(np.float32)
        stacked_lab[i] = lab

    # Extract L channel (luminance) only: shape (num_frames, height, width)
    luminance = stacked_lab[:, :, :, 0]

    # Compute per-pixel median luminance across frames
    # Result shape: (height, width)
    median_luminance = np.median(luminance, axis=0)

    # For each pixel, compute distance to median luminance in each frame
    # distances shape: (num_frames, height, width)
    distances = np.abs(luminance - median_luminance)

    # Find the frame with minimum luminance distance for each pixel
    # Result shape: (height, width)
    closest_frame_idx = np.argmin(distances, axis=0)

    # Build output by picking BGR pixels from the closest frame
    # For each (h, w) position, use the BGR pixel from frames[closest_frame_idx[h, w]]
    # Vectorized approach: use take_along_axis to select pixels in O(1) operation
    height, width = closest_frame_idx.shape
    c = stacked_bgr.shape[3]

    # Expand closest_frame_idx to (height, width, 3) for broadcasting across channels
    frame_indices_expanded = np.expand_dims(closest_frame_idx, axis=-1)  # (height, width, 1)
    frame_indices_expanded = np.broadcast_to(frame_indices_expanded, (height, width, c))  # (height, width, 3)

    # Use take_along_axis to select BGR pixels along frame axis
    # stacked_bgr is (num_frames, height, width, 3), indices are (height, width, 3)
    # Result shape: (1, height, width, 3)
    output = np.take_along_axis(stacked_bgr, frame_indices_expanded[np.newaxis, ...], axis=0)
    output = output.squeeze(axis=0)  # (height, width, 3)

    # Convert back to uint8
    return np.clip(output, 0, 255).astype(np.uint8)
