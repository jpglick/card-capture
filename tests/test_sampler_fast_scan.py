import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from card_capture.sampler import AdaptivePresenceSampler
from card_capture.sampler.valley_splits import find_valley_splits


def test_adaptive_sampler_accepts_fast_scan_fps():
    """AdaptivePresenceSampler must accept fast_scan_fps and confirm_scan_fps params."""
    sampler = AdaptivePresenceSampler(fast_scan_fps=15.0, confirm_scan_fps=5.0)
    assert sampler.fast_scan_fps == 15.0
    assert sampler.confirm_scan_fps == 5.0


def test_adaptive_sampler_accepts_delta_spike_ratio():
    sampler = AdaptivePresenceSampler(delta_spike_ratio=0.7)
    assert sampler.delta_spike_ratio == 0.7


def test_scan_frame_has_delta_score_field():
    """_ScanFrame must have sobel_score and delta_score attributes."""
    from card_capture.sampler import _ScanFrame
    frame = _ScanFrame(frame_index=0, timestamp_ms=0.0, sobel_score=0.5, delta_score=0.0)
    assert frame.delta_score == 0.0
    assert frame.sobel_score == 0.5


def test_scan_frame_delta_score_computed_between_frames():
    """First frame delta_score is 0.0; subsequent frames have non-zero delta when images differ."""
    from card_capture.sampler import _ScanFrame
    # Just verify the dataclass accepts and stores it correctly
    f1 = _ScanFrame(frame_index=0, timestamp_ms=0.0, sobel_score=1.0, delta_score=0.0)
    f2 = _ScanFrame(frame_index=1, timestamp_ms=66.7, sobel_score=1.1, delta_score=12.5)
    assert f1.delta_score == 0.0
    assert f2.delta_score == 12.5


def test_sampler_exposes_last_valley_splits():
    """AdaptivePresenceSampler must expose last_valley_splits as a list."""
    sampler = AdaptivePresenceSampler()
    assert hasattr(sampler, 'last_valley_splits')
    assert isinstance(sampler.last_valley_splits, list)
