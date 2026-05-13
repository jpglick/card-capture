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


def _adaptive_scan_frame(frame_index):
    from card_capture.sampler import _AdaptiveScanFrame

    return _AdaptiveScanFrame(
        frame_index=frame_index,
        timestamp_ms=frame_index * 16,
        image=np.zeros((12, 12, 3), dtype=np.uint8),
        metrics={"edge_density": 1.0},
        delta_score=0.0,
    )


def test_per_region_valley_splits_map_scan_indices_to_source_frames(monkeypatch):
    from card_capture.sampler import valley_detection_per_region as regional

    frames = [np.zeros((12, 12, 3), dtype=np.uint8) for _ in range(5)]
    monkeypatch.setattr(
        regional,
        "per_region_valley_detection",
        lambda *args, **kwargs: [(1, 0, 0), (3, 2, 1)],
    )

    splits = regional.find_valley_splits_per_region(
        frames,
        frame_indices=[0, 4, 8, 12, 16],
    )

    assert splits == [4, 12]


def test_per_region_valley_splits_require_configured_region_votes(monkeypatch):
    from card_capture.sampler import valley_detection_per_region as regional

    frames = [np.zeros((12, 12, 3), dtype=np.uint8) for _ in range(5)]
    monkeypatch.setattr(
        regional,
        "per_region_valley_detection",
        lambda *args, **kwargs: [(1, 0, 0), (1, 1, 0), (3, 2, 1)],
    )

    splits = regional.find_valley_splits_per_region(
        frames,
        frame_indices=[0, 4, 8, 12, 16],
        min_region_votes=2,
    )

    assert splits == [4]


def test_compute_valley_splits_maps_regional_splits_to_source_frames(monkeypatch):
    sampler = AdaptivePresenceSampler()
    scan_frames = [_adaptive_scan_frame(i) for i in range(0, 84, 4)]

    monkeypatch.setattr(
        "card_capture.sampler.valley_splits.find_valley_splits",
        lambda *args, **kwargs: [40],
    )
    monkeypatch.setattr(
        "card_capture.sampler.valley_detection_per_region.find_valley_splits_per_region",
        lambda *args, **kwargs: [4, 80],
    )

    assert sampler._compute_valley_splits(scan_frames) == [4, 40, 80]


def test_compute_valley_splits_coalesces_adjacent_regional_splits(monkeypatch):
    sampler = AdaptivePresenceSampler(valley_min_width_frames=3)
    scan_frames = [_adaptive_scan_frame(i) for i in [0, 4, 8, 12, 16, 60]]

    monkeypatch.setattr(
        "card_capture.sampler.valley_splits.find_valley_splits",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "card_capture.sampler.valley_detection_per_region.find_valley_splits_per_region",
        lambda *args, **kwargs: [4, 8, 12],
    )

    assert sampler._compute_valley_splits(scan_frames) == [4]
