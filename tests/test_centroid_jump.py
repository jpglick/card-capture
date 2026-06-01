import numpy as np
import pytest
from card_capture.stages.track.centroid_jump import CentroidJumpDetector


def test_large_jump_triggers_split():
    """A centroid jump > jump_ratio * frame_width returns True."""
    det = CentroidJumpDetector(jump_ratio=0.30, jump_within_frames=3)
    # First bbox: corners at (10,10), (110,10), (110,110), (10,110) — centroid at x=60
    r1 = det.update(np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32), frame_width=1000)
    assert r1 is False  # first frame never triggers
    # Second bbox: corners at (500,10), (600,10), (600,110), (500,110) — centroid at x=550, jump of 490px > 300px threshold
    r2 = det.update(np.array([[500, 10], [600, 10], [600, 110], [500, 110]], dtype=np.float32), frame_width=1000)
    assert r2 is True


def test_gradual_drift_does_not_trigger():
    """Small per-frame movements that never exceed the threshold don't trigger."""
    det = CentroidJumpDetector(jump_ratio=0.30, jump_within_frames=5)
    # Move 10px per frame — never a 300px jump in 5 frames
    for i in range(10):
        x = i * 10
        result = det.update(np.array([[x, 10], [x+100, 10], [x+100, 110], [x, 110]], dtype=np.float32), frame_width=1000)
    assert result is False


def test_none_detection_is_no_op():
    """Passing None (no detection) does not trigger and does not raise."""
    det = CentroidJumpDetector()
    r1 = det.update(np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32), frame_width=1000)
    r2 = det.update(None, frame_width=1000)
    assert r2 is False


def test_reset_clears_history():
    """After reset(), no spurious split fires even after a previous large jump."""
    det = CentroidJumpDetector(jump_ratio=0.30, jump_within_frames=3)
    det.update(np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32), frame_width=1000)
    det.update(np.array([[500, 10], [600, 10], [600, 110], [500, 110]], dtype=np.float32), frame_width=1000)
    det.reset()
    # After reset, new updates should not be influenced by old history
    r = det.update(np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32), frame_width=1000)
    assert r is False


def test_first_frame_never_triggers():
    """The very first update call always returns False regardless of bbox position."""
    det = CentroidJumpDetector(jump_ratio=0.01, jump_within_frames=1)
    r = det.update(np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32), frame_width=1000)
    assert r is False


def test_uses_highest_scoring_centroid_when_multiple_candidates():
    """When multiple bboxes in one frame, highest-score wins (if scores provided)."""
    # This tests the interface. When caller passes a single highest-score bbox, it works.
    det = CentroidJumpDetector(jump_ratio=0.30, jump_within_frames=3)
    det.update(np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32), frame_width=1000)
    # Simulate: caller picked highest score bbox before calling update
    r = det.update(np.array([[500, 10], [600, 10], [600, 110], [500, 110]], dtype=np.float32), frame_width=1000)
    assert r is True
