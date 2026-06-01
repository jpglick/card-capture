"""ECC (Enhanced Correlation Coefficient) based frame registration for alignment before fusion.

This module provides functions to align video frames to a reference frame using
OpenCV's ECC algorithm. This eliminates ghosting artifacts caused by corner
detector jitter (typically ±1-3 pixels) before performing median fusion.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def compute_ecc_warp(
    reference_frame: np.ndarray,
    target_frame: np.ndarray,
    motion_model: int = cv2.MOTION_TRANSLATION
) -> np.ndarray:
    """
    Compute the ECC warp matrix to align a target frame to a reference frame.

    Args:
        reference_frame: Reference frame (BGR or grayscale)
        target_frame: Frame to be registered to the reference
        motion_model: cv2.MOTION_* constant (default: MOTION_TRANSLATION for 2px jitter)

    Returns:
        warp_matrix: 2x3 affine transformation matrix. If convergence fails,
                    returns identity matrix for translation.
    """
    # Convert frames to grayscale
    if reference_frame.ndim == 3:
        ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = reference_frame.copy()

    if target_frame.ndim == 3:
        target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    else:
        target_gray = target_frame.copy()

    # Initialize warp matrix for translation (identity + zero translation vector)
    if motion_model == cv2.MOTION_TRANSLATION:
        warp_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]], dtype=np.float32)
    elif motion_model == cv2.MOTION_AFFINE:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif motion_model == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, dtype=np.float32)
    else:
        # Default to translation for unknown motion models
        warp_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]], dtype=np.float32)

    # Criteria for ECC convergence
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 1e-3)

    try:
        # Find the geometric transformation that aligns target_gray to ref_gray.
        # cv2.findTransformECC(template, image, ...) finds W such that warpAffine(image, W) aligns with template.
        # So we call findTransformECC(target, reference) to find W such that warpAffine(target, W) aligns with reference.
        cc, warp_matrix = cv2.findTransformECC(
            target_gray, ref_gray, warp_matrix, motion_model, criteria
        )
    except cv2.error as e:
        # If convergence fails, return identity matrix to avoid misalignment
        print(f"ECC convergence failed: {e}. Using identity matrix.")
        if motion_model == cv2.MOTION_TRANSLATION:
            warp_matrix = np.array([[1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0]], dtype=np.float32)
        elif motion_model == cv2.MOTION_AFFINE:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(3, dtype=np.float32)

    return warp_matrix


def register_frames_via_ecc(
    frames: List[np.ndarray],
    reference_index: int = 0
) -> List[np.ndarray]:
    """
    Register all frames to a reference frame using ECC alignment.

    Args:
        frames: List of frames (BGR or grayscale) to register
        reference_index: Index of the reference frame in the list (default: 0)

    Returns:
        List of aligned frames with the reference frame at reference_index unchanged.
        If frames list has fewer than 2 frames, returns the original frames unchanged.
    """
    if len(frames) < 2:
        return frames

    reference_frame = frames[reference_index]
    aligned_frames = []

    for i, frame in enumerate(frames):
        if i == reference_index:
            # Reference frame stays unchanged
            aligned_frames.append(frame)
        else:
            # Compute warp matrix for this frame
            warp_matrix = compute_ecc_warp(reference_frame, frame)

            # Apply the warp transformation
            if frame.ndim == 3:
                h, w = frame.shape[:2]
                aligned_frame = cv2.warpAffine(
                    frame, warp_matrix, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
            else:
                h, w = frame.shape[:2]
                aligned_frame = cv2.warpAffine(
                    frame, warp_matrix, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )

            aligned_frames.append(aligned_frame)

    return aligned_frames
