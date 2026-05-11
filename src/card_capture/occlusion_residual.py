"""Per-tile occlusion residual detection via running median.

Detects interior occlusions (fingers, stickers, lens flare, foil shifts)
by computing per-tile median residuals from prior frames.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def compute_occlusion_residual_score(
    current_frame: np.ndarray,
    prior_frames: Sequence[np.ndarray],
    grid_rows: int = 5,
    grid_cols: int = 7,
    residual_threshold: float = 0.08,
) -> float:
    """Detect occlusions via per-tile median residual analysis.

    Args:
        current_frame: Current grayscale frame (h, w) as uint8.
        prior_frames: Prior grayscale frames (list of (h, w) uint8 arrays).
        grid_rows: Number of tile rows (default 5).
        grid_cols: Number of tile columns (default 7).
        residual_threshold: High-residual fraction of max per-tile value for detection.

    Returns:
        Score in [0.0, 1.0] where 1.0 = no occlusion, 0.0 = severe occlusion.

    Strategy:
        1. Tile current frame and prior frames into (grid_rows, grid_cols) grid.
        2. Compute per-tile median from prior frames.
        3. Compute per-tile residual: |current - median| normalized by tile's range.
        4. Mark tiles with high mean residual (>threshold), not just pixel count.
        5. Use connected-component analysis to find high-residual blobs.
        6. If blob covers <30% of tiles = localized occlusion, penalize.
        7. Return score: 1.0 (no occlusion) → 0.0 (severe occlusion).
    """
    if not prior_frames:
        # No prior frames; assume no occlusion (neutral).
        return 1.0

    h, w = current_frame.shape[:2]
    tile_h = h // grid_rows
    tile_w = w // grid_cols

    if tile_h <= 0 or tile_w <= 0:
        # Frame too small to tile; neutral.
        return 1.0

    # Ensure current frame is float [0, 1] for easier math
    current_float = current_frame.astype(np.float32) / 255.0

    # Build prior frames as float, clipping to same tile size
    prior_float = []
    for pf in prior_frames:
        if pf.shape != current_frame.shape:
            # Resize prior frame to match current if needed
            pf = cv2.resize(pf, (w, h), interpolation=cv2.INTER_LINEAR)
        prior_float.append(pf.astype(np.float32) / 255.0)

    if not prior_float:
        return 1.0

    # Compute per-tile median from prior frames
    prior_stack = np.stack(prior_float, axis=0)  # (num_priors, h, w)

    # First pass: compute all tile residuals to determine adaptive threshold
    tile_residuals = np.zeros((grid_rows, grid_cols))

    for i in range(grid_rows):
        for j in range(grid_cols):
            y_start = i * tile_h
            y_end = (i + 1) * tile_h if i < grid_rows - 1 else h
            x_start = j * tile_w
            x_end = (j + 1) * tile_w if j < grid_cols - 1 else w

            # Extract tile from current and prior frames
            current_tile = current_float[y_start:y_end, x_start:x_end]
            prior_tiles = prior_stack[:, y_start:y_end, x_start:x_end]  # (num_priors, th, tw)

            # Compute per-pixel median from prior frames
            median_tile = np.median(prior_tiles, axis=0)  # (th, tw)

            # Compute residual: |current - median|
            residual = np.abs(current_tile - median_tile)

            # Normalize by tile's intensity range (avoid division by zero)
            tile_max = np.max(np.abs(median_tile))
            if tile_max < 1e-3:
                # Very dark tile; use raw residual
                tile_residuals[i, j] = residual.mean()
            else:
                # Normalized residual
                norm_residual = residual / tile_max
                tile_residuals[i, j] = norm_residual.mean()

    # Adaptive threshold: mark tiles that are significantly above median residual
    # This prevents false positives when all frames are noisy
    global_median_residual = np.median(tile_residuals[tile_residuals > 0]) if np.any(tile_residuals > 0) else 0
    # Use 3.0x multiplier to distinguish occlusions from baseline frame differences
    # Clean frames have baseline residual ~0.08, fingertip adds ~0.11, but random frames have 0.20
    adaptive_threshold = max(residual_threshold, global_median_residual * 3.0)

    # Second pass: mark high-residual tiles
    high_residual_mask = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
    for i in range(grid_rows):
        for j in range(grid_cols):
            if tile_residuals[i, j] > adaptive_threshold:
                high_residual_mask[i, j] = 1

    # Connected-component analysis on high-residual tiles
    num_labels, labels = cv2.connectedComponents(
        high_residual_mask, connectivity=8
    )

    if num_labels <= 1:
        # No high-residual tiles (label 0 is background).
        return 1.0

    # Count tiles per component
    total_tiles = grid_rows * grid_cols

    # Find the largest high-residual blob
    max_blob_tiles = 0
    for label_id in range(1, num_labels):
        component_tiles = np.sum(labels == label_id)
        max_blob_tiles = max(max_blob_tiles, component_tiles)

    # Penalize if any blob is detected (any high-residual region).
    # Strategy: blob size determines penalty magnitude.
    # - Very small blobs (<5% of frame): minimal penalty (~0.1)
    # - Medium blobs (5-25%): moderate penalty
    # - Large blobs (>25%): severe penalty

    blob_fraction = max_blob_tiles / total_tiles

    # Penalty curve: blob_fraction → penalty
    # At blob_fraction=0.05 (minimal), penalty ~0.1 → score ~0.9
    # At blob_fraction=0.15 (medium), penalty ~0.3 → score ~0.7
    # At blob_fraction=0.30 (large), penalty ~0.6 → score ~0.4
    # At blob_fraction=0.50 (very large), penalty ~0.95 → score ~0.05

    if blob_fraction > 0.0:
        # Use linear penalty: penalty = blob_fraction
        # This directly maps blob coverage to penalty (e.g., 40% coverage = 0.4 penalty)
        penalty = blob_fraction
        occlusion_score = max(0.0, 1.0 - penalty)
        return float(occlusion_score)

    return 1.0
