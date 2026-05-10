import numpy as np

from card_capture.adaptive_gap import compute_session_gap_frames, GapDistribution


def test_compute_gap_returns_p95_plus_buffer():
    # Inter-window gaps in frames: mostly 5, with a few outliers
    gaps = [3, 4, 5, 5, 6, 5, 7, 5, 6, 4, 30]  # P95 around the high tail
    result = compute_session_gap_frames(gaps, fps=30.0)
    assert isinstance(result, GapDistribution)
    assert result.recommended_gap_frames >= 7  # at least P95 of the typical body
    assert result.recommended_gap_frames <= 90  # capped at 3s @ 30fps
    assert result.p50_frames == 5


def test_floor_minimum_when_gaps_tiny():
    gaps = [1, 1, 1, 2, 1]  # all very small
    result = compute_session_gap_frames(gaps, fps=30.0)
    # 0.5s floor at 30fps = 15 frames
    assert result.recommended_gap_frames >= 15


def test_cap_at_three_seconds():
    gaps = [200, 250, 300, 350, 400]  # huge
    result = compute_session_gap_frames(gaps, fps=30.0)
    # 3s cap at 30fps = 90 frames
    assert result.recommended_gap_frames == 90


def test_empty_input_returns_default():
    result = compute_session_gap_frames([], fps=30.0)
    assert result.recommended_gap_frames == 15  # 0.5s default at 30fps
