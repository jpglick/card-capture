from __future__ import annotations
import builtins
import numpy as np
from card_capture.ingestion import (
    FrameTriageFilter,
    _decord_available,
    _resolve_reader_backend,
    RollingWindowTriage
)

def test_frame_triage_filter_rejects_empty_frame() -> None:
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    triage = FrameTriageFilter(empty_ratio_threshold=0.95, variance_threshold=1.0)
    accepted, metrics = triage.evaluate(frame)
    assert accepted is False
    assert metrics["empty_ratio"] >= 0.95

def test_frame_triage_filter_accepts_textured_frame() -> None:
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    triage = FrameTriageFilter(empty_ratio_threshold=0.95, variance_threshold=100.0)
    accepted, metrics = triage.evaluate(frame)
    assert accepted is True
    assert metrics["variance"] > 100.0

def test_rolling_window_triage_cold_start():
    triage = RollingWindowTriage(window_size=10, keep_percentile=0.5)
    # Frames 0-8 should all be accepted (wide funnel during cold start)
    for i in range(9):
        assert triage.evaluate_score(i, 1.0) is True

def test_rolling_window_triage_sliding():
    triage = RollingWindowTriage(window_size=10, keep_percentile=0.5)
    # Fill window
    for i in range(10):
        triage.evaluate_score(i, float(i))
    
    # Now it should start filtering
    # Current buffer: [(0,0), (1,1), ... (9,9)]. keep_count = 5. Top are 5,6,7,8,9.
    assert triage.evaluate_score(10, 10.0) is True # 10 is the new top
    assert triage.evaluate_score(11, 0.0) is False # 0 is at the bottom
