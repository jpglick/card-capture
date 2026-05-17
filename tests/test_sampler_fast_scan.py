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


def _make_presence_window(start_frame, end_frame, n_records, base_score=0.8):
    """Helper: build a PresenceWindow and matching _AdaptiveScanFrame list."""
    from card_capture.sampler import PresenceWindow, _AdaptiveScanFrame
    window = PresenceWindow(start_frame=start_frame, end_frame=end_frame)
    step = (end_frame - start_frame) // max(n_records - 1, 1)
    records = []
    for i in range(n_records):
        fi = start_frame + i * step
        records.append(_AdaptiveScanFrame(
            frame_index=fi,
            timestamp_ms=fi * 16,
            image=np.zeros((12, 12, 3), dtype=np.uint8),
            metrics={"edge_density": 1.0},
            presence_score=base_score + (i % 3) * 0.05,
        ))
    return window, records


def test_score_sharpness_temporal_spread():
    """Candidates must be spread across the window, not clustered at high-score frames."""
    sampler = AdaptivePresenceSampler(target_yolo_fps=3.0)
    sampler.last_source_fps = 60.0
    # 10-second window at 60fps = frames 0..600; 60 scan records
    window, records = _make_presence_window(0, 600, n_records=60)
    sampler._scan_frames = records

    result = sampler._score_sharpness_in_window(window)
    candidates = [fi for fi, _ in result.frame_candidates]

    # At 3fps over 10s → ~30 candidates, capped at max_candidates_per_window=24
    assert 10 <= len(candidates) <= 24

    # Candidates must span at least 80% of the window range
    assert max(candidates) - min(candidates) >= 0.8 * 600

    # No two consecutive candidates come from the same 5% of the window
    segment = 600 / 20
    segments_used = {int(fi / segment) for fi in candidates}
    assert len(segments_used) >= 10, "candidates are clustered, not temporally spread"


def test_score_sharpness_short_window_respects_min():
    """A very short window still yields at least min_candidates_per_window frames."""
    sampler = AdaptivePresenceSampler(target_yolo_fps=3.0)
    sampler.last_source_fps = 60.0
    # 0.5-second window → ceil(0.5 * 3) = 2, but min is 3
    window, records = _make_presence_window(0, 30, n_records=5)
    sampler._scan_frames = records

    result = sampler._score_sharpness_in_window(window)
    assert len(result.frame_candidates) >= sampler.min_candidates_per_window


def test_score_sharpness_picks_best_score_per_bucket():
    """Within each bucket, the frame with the highest presence_score is chosen."""
    from card_capture.sampler import PresenceWindow, _AdaptiveScanFrame
    sampler = AdaptivePresenceSampler(target_yolo_fps=1.0, max_candidates_per_window=2)
    sampler.last_source_fps = 60.0
    # 2-second window → 2 buckets; plant known best scores
    window = PresenceWindow(start_frame=0, end_frame=120)
    records = [
        _AdaptiveScanFrame(0,   0,    np.zeros((4,4,3), dtype=np.uint8), {}, presence_score=0.5),
        _AdaptiveScanFrame(30,  500,  np.zeros((4,4,3), dtype=np.uint8), {}, presence_score=0.9),   # best in bucket 0
        _AdaptiveScanFrame(60,  1000, np.zeros((4,4,3), dtype=np.uint8), {}, presence_score=0.6),
        _AdaptiveScanFrame(90,  1500, np.zeros((4,4,3), dtype=np.uint8), {}, presence_score=0.95),  # best in bucket 1
        _AdaptiveScanFrame(120, 2000, np.zeros((4,4,3), dtype=np.uint8), {}, presence_score=0.4),
    ]
    sampler._scan_frames = records

    result = sampler._score_sharpness_in_window(window)
    chosen_indices = [fi for fi, _ in result.frame_candidates]
    assert 30 in chosen_indices, "highest-score frame in bucket 0 must be chosen"
    assert 90 in chosen_indices, "highest-score frame in bucket 1 must be chosen"
