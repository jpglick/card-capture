"""Per-region valley detection for multi-card swaps.

Tiles the proxy frame into a 3×3 grid and detects valleys (70% edge density drop)
independently in each tile. This detects swaps on opposite sides (e.g., card A leaves
left while card B enters right) that global Sobel detection misses.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import cv2


def per_region_valley_detection(
    frames: list[np.ndarray],
    grid_rows: int = 3,
    grid_cols: int = 3,
    valley_drop_ratio: float = 0.70,
    min_valley_width_frames: int = 3,
) -> list[tuple[int, int, int]]:
    """Detect valleys per tile in a frame sequence.

    For each frame, tiles it into grid_rows × grid_cols cells and computes the
    Sobel edge density per tile. Then detects valleys (70% drop from preceding peak)
    in each tile independently. Valleys must persist >= min_valley_width_frames.

    Args:
        frames: List of frames (H × W or H × W × 3 arrays).
        grid_rows: Number of rows in the grid (default 3).
        grid_cols: Number of columns in the grid (default 3).
        valley_drop_ratio: Drop ratio threshold for valley detection (default 0.70 = 70%).
        min_valley_width_frames: Minimum valley width in frames (default 3).

    Returns:
        List of (frame_idx, row, col) tuples where valleys are detected.
    """
    if not frames:
        return []

    # Ensure frames are grayscale
    gray_frames = []
    for frame in frames:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        gray_frames.append(gray)

    # Compute edge density per tile per frame
    tile_scores = _compute_tile_edge_densities(gray_frames, grid_rows, grid_cols)

    # Detect valleys in each tile
    valleys = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            tile_key = (row, col)
            tile_density_sequence = tile_scores[tile_key]

            # Apply valley detection to this tile's sequence
            valley_frames = _find_valleys_in_sequence(
                tile_density_sequence,
                valley_drop_ratio=valley_drop_ratio,
                min_valley_width_frames=min_valley_width_frames,
            )

            # Add (frame_idx, row, col) tuples to result
            for frame_idx in valley_frames:
                valleys.append((frame_idx, row, col))

    return sorted(valleys)


def _compute_tile_edge_densities(
    gray_frames: list[np.ndarray],
    grid_rows: int,
    grid_cols: int,
) -> dict[tuple[int, int], list[float]]:
    """Compute Sobel edge density for each tile across all frames.

    Args:
        gray_frames: List of grayscale frames.
        grid_rows: Number of rows in the grid.
        grid_cols: Number of columns in the grid.

    Returns:
        Dictionary mapping (row, col) to list of edge densities across frames.
    """
    tile_scores: dict[tuple[int, int], list[float]] = {}
    for row in range(grid_rows):
        for col in range(grid_cols):
            tile_scores[(row, col)] = []

    for frame in gray_frames:
        h, w = frame.shape[:2]
        tile_height = h // grid_rows
        tile_width = w // grid_cols

        # Compute Sobel for the entire frame (more efficient)
        sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Extract density per tile
        for row in range(grid_rows):
            for col in range(grid_cols):
                y_start = row * tile_height
                y_end = (row + 1) * tile_height if row < grid_rows - 1 else h
                x_start = col * tile_width
                x_end = (col + 1) * tile_width if col < grid_cols - 1 else w

                tile_sobel = sobel_mag[y_start:y_end, x_start:x_end]
                # Edge density: proportion of pixels with significant edge (>50)
                edge_count = np.sum(tile_sobel > 50)
                tile_area = tile_sobel.size
                density = float(edge_count) / float(tile_area) if tile_area > 0 else 0.0
                tile_scores[(row, col)].append(density)

    return tile_scores


def _find_valleys_in_sequence(
    scores: list[float],
    valley_drop_ratio: float = 0.70,
    min_valley_width_frames: int = 3,
) -> list[int]:
    """Find frames where a valley (drop and recovery) occurs in a score sequence.

    Args:
        scores: List of scores (edge densities).
        valley_drop_ratio: Threshold for drop ratio (e.g., 0.70 = 70% drop).
        min_valley_width_frames: Minimum valley width in frames.

    Returns:
        List of frame indices where valleys are detected.
    """
    if len(scores) < min_valley_width_frames:
        return []

    valley_frames = []
    peak = scores[0]
    valley_start: int | None = None

    for i, score in enumerate(scores):
        if score > peak:
            peak = score

        # Valley condition: drop >= valley_drop_ratio from peak
        in_valley = score < peak * (1.0 - valley_drop_ratio)

        if in_valley:
            if valley_start is None:
                valley_start = i
        else:
            if valley_start is not None:
                width = i - valley_start
                if width >= min_valley_width_frames:
                    # Find the minimum within the valley
                    min_idx = valley_start + int(np.argmin(scores[valley_start:i]))
                    valley_frames.append(min_idx)
                valley_start = None

    # Handle valley extending to end of signal
    if valley_start is not None:
        width = len(scores) - valley_start
        if width >= min_valley_width_frames:
            min_idx = valley_start + int(np.argmin(scores[valley_start:]))
            valley_frames.append(min_idx)

    return valley_frames


def find_valley_splits_per_region(
    frames: list[np.ndarray],
    grid_rows: int = 3,
    grid_cols: int = 3,
    valley_drop_ratio: float = 0.70,
    min_valley_width_frames: int = 3,
    frame_indices: list[int] | None = None,
    min_region_votes: int = 1,
) -> list[int]:
    """Find unique split frames where multi-region valleys are detected.

    Aggregates per-region valleys by frame index and returns sorted unique frames.

    Args:
        frames: List of frames (H × W or H × W × 3 arrays).
        grid_rows: Number of rows in the grid (default 3).
        grid_cols: Number of columns in the grid (default 3).
        valley_drop_ratio: Drop ratio threshold for valley detection (default 0.70).
        min_valley_width_frames: Minimum valley width in frames (default 3).
        frame_indices: Optional source-frame indices corresponding to ``frames``.
            When omitted, returns scan-sequence indices for backward compatibility.
        min_region_votes: Minimum number of grid regions that must vote for
            the same split frame. The default preserves historical behavior.

    Returns:
        Sorted list of unique split frame indices where valleys are detected.
    """
    if frame_indices is not None and len(frame_indices) != len(frames):
        raise ValueError(
            f"frame_indices length ({len(frame_indices)}) must equal frames length ({len(frames)})"
        )

    valleys = per_region_valley_detection(
        frames,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        valley_drop_ratio=valley_drop_ratio,
        min_valley_width_frames=min_valley_width_frames,
    )

    min_votes = max(1, int(min_region_votes))
    vote_counts = Counter(frame_idx for frame_idx, _, _ in valleys)

    # Extract unique frame indices
    if frame_indices is None:
        split_frames = sorted(
            frame_idx for frame_idx, votes in vote_counts.items() if votes >= min_votes
        )
    else:
        split_frames = sorted(
            frame_indices[frame_idx]
            for frame_idx, votes in vote_counts.items()
            if votes >= min_votes
        )
    return split_frames
