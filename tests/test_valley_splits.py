from card_capture.stages.sample.sampler.valley_splits import find_valley_splits


def test_flat_signal_no_splits():
    scores = [1.0, 1.0, 1.0, 1.0, 1.0]
    deltas = [0.0] * 5
    frames = list(range(5))
    assert find_valley_splits(scores, deltas, frames) == []


def test_single_qualified_valley_returns_split_frame():
    # Peak at 1.2, drops to 0.3-0.4 (>40% drop) for 2 frames, recovers
    scores = [1.0, 1.2, 0.4, 0.3, 1.1, 1.3]
    frames = [10, 11, 12, 13, 14, 15]
    deltas = [0.0] * 6
    splits = find_valley_splits(scores, deltas, frames, valley_drop_ratio=0.4, valley_min_width_frames=2)
    assert splits == [13]  # frame at minimum of valley


def test_shallow_valley_no_split():
    # Drop of only 10% — below threshold
    scores = [1.0, 1.1, 0.9, 1.0]
    frames = [1, 2, 3, 4]
    deltas = [0.0] * 4
    assert find_valley_splits(scores, deltas, frames, valley_drop_ratio=0.4, valley_min_width_frames=2) == []


def test_delta_spike_triggers_split_without_sobel_valley():
    # Two similarly edge-dense cards — Sobel stays flat, pixel delta spikes on swap
    scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    deltas = [0.02, 0.03, 0.85, 0.03, 0.02, 0.02]  # big spike at index 2 = frame 12
    frames = [10, 11, 12, 13, 14, 15]
    splits = find_valley_splits(scores, deltas, frames, delta_spike_ratio=0.5)
    assert 12 in splits


def test_combined_signal_no_double_split():
    # Sobel valley and delta spike coincide — should produce only one split point
    scores = [1.2, 1.1, 0.3, 0.3, 1.0, 1.1]
    deltas = [0.02, 0.02, 0.80, 0.04, 0.02, 0.02]
    frames = [10, 11, 12, 13, 14, 15]
    splits = find_valley_splits(
        scores, deltas, frames,
        valley_drop_ratio=0.4, valley_min_width_frames=2, delta_spike_ratio=0.5
    )
    assert len(splits) == 1


def test_empty_inputs_return_empty():
    assert find_valley_splits([], [], []) == []


def test_result_is_sorted():
    # Two separate valleys — result must be sorted
    scores = [1.0, 0.2, 0.2, 1.0, 0.2, 0.2, 1.0]
    deltas = [0.0] * 7
    frames = [0, 1, 2, 3, 4, 5, 6]
    splits = find_valley_splits(scores, deltas, frames, valley_drop_ratio=0.4, valley_min_width_frames=2)
    assert splits == sorted(splits)
    assert len(splits) == 2


def test_trailing_valley_at_end_of_signal():
    # Valley at end of signal (no recovery) — should still split if wide enough
    scores = [1.0, 1.2, 0.3, 0.25, 0.2]
    deltas = [0.0] * 5
    frames = [0, 1, 2, 3, 4]
    splits = find_valley_splits(scores, deltas, frames, valley_drop_ratio=0.4, valley_min_width_frames=3)
    assert len(splits) == 1
    assert splits[0] == 4  # minimum is at last frame


def test_mismatched_lengths_raise_value_error():
    import pytest
    with pytest.raises(ValueError):
        find_valley_splits([1.0, 1.0], [0.0], [0, 1])  # delta_scores wrong length
