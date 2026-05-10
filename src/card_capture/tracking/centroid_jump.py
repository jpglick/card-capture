from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


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

    def update(self, bbox_xyxy: Optional[np.ndarray], frame_width: int) -> bool:
        """Returns True if a centroid jump exceeding threshold is detected.

        Pass None if no detection in this frame (treated as no-op — returns False).
        The caller is responsible for passing the highest-scoring candidate's bbox
        when multiple detections are present.
        """
        if bbox_xyxy is None:
            return False

        cx = float((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0)
        cy = float((bbox_xyxy[1] + bbox_xyxy[3]) / 2.0)
        centroid = (cx, cy)

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
