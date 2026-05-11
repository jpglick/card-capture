from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np


def centroid_from_obb(obb_corners: np.ndarray) -> Tuple[float, float]:
    """Compute the centroid (mean of corners) of an OBB (oriented bounding box).

    Args:
        obb_corners: Array of shape (4, 2) with corner coordinates (x, y).
                     Order doesn't matter; the centroid is the mean of all 4 points.

    Returns:
        Tuple of (cx, cy) representing the centroid.

    Example:
        >>> corners = np.array([(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> centroid_from_obb(corners)
        (50.0, 50.0)
    """
    obb_array = np.asarray(obb_corners, dtype=np.float64)
    cx = float(obb_array[:, 0].mean())
    cy = float(obb_array[:, 1].mean())
    return (cx, cy)


class CentroidJumpDetector:
    """Detects sudden spatial jumps in card centroid position.

    Maintains a rolling history of centroids. Returns True if the maximum
    centroid displacement over the last jump_within_frames frames exceeds
    jump_ratio * frame_width.
    """

    def __init__(
        self,
        jump_ratio: float = 0.30,
        jump_within_frames: int = 3,
    ) -> None:
        self.jump_ratio = jump_ratio
        self.jump_within_frames = jump_within_frames
        self._history: deque[tuple[float, float]] = deque(maxlen=jump_within_frames)

    def update(self, obb_corners: Optional[np.ndarray], frame_width: int) -> bool:
        """Returns True if a centroid jump exceeding threshold is detected.

        Pass None if no detection in this frame (treated as no-op — returns False).
        The caller is responsible for passing the highest-scoring candidate's OBB corners
        when multiple detections are present.

        Args:
            obb_corners: Array of shape (4, 2) with corner coordinates (x, y),
                        or None if no detection in this frame.
            frame_width: Width of the frame in pixels, used to normalize jump threshold.

        Returns:
            True if a centroid jump exceeding the threshold is detected.
        """
        if obb_corners is None:
            return False

        centroid = centroid_from_obb(obb_corners)
        cx, cy = centroid

        if not self._history:
            self._history.append(centroid)
            return False

        threshold = self.jump_ratio * frame_width
        jumped = any(
            abs(cx - hx) > threshold
            for hx, _hy in self._history
        )

        self._history.append(centroid)
        return jumped

    def reset(self) -> None:
        """Call on session reset to clear centroid history."""
        self._history.clear()
