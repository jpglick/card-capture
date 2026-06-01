import numpy as np

from card_capture.tracking.centroid_jump import CentroidJumpDetector


def _box(cx, cy, s=50):
    return np.array(
        [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)],
        dtype=np.float32,
    )


def test_detects_vertical_jump():
    detector = CentroidJumpDetector(jump_ratio=0.30, jump_within_frames=3)
    assert detector.update(_box(500, 100), frame_width=1000) is False
    assert detector.update(_box(510, 800), frame_width=1000) is True

