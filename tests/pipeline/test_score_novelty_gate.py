import pytest


def _gate(scores):
    from pipeline.steps.score import _novelty_gate_useful
    return _novelty_gate_useful(scores)


def test_bimodal_distribution_activates_gate():
    """Stand-style video: low-novelty stand detections + high-novelty card detections."""
    scores = [0.05, 0.08, 0.85, 0.90, 0.92]
    assert _gate(scores) is True


def test_all_high_scores_disables_gate():
    """Hand-held video: everything is novel vs background — std too low."""
    scores = [0.82, 0.88, 0.91, 0.87, 0.85]
    assert _gate(scores) is False


def test_too_few_samples_disables_gate():
    """Fewer than 5 detections — not enough data to judge distribution."""
    scores = [0.05, 0.90]
    assert _gate(scores) is False


def test_high_std_but_min_not_low_enough_disables_gate():
    """Spread exists but nothing scores below 0.35 — no background-like detections."""
    scores = [0.40, 0.90, 0.91, 0.40, 0.88]
    assert _gate(scores) is False


def test_empty_scores_disables_gate():
    """No detections at all — gate must not fire."""
    assert _gate([]) is False


def test_exactly_five_samples_bimodal_activates():
    """Boundary: exactly 5 samples with bimodal distribution."""
    scores = [0.10, 0.12, 0.80, 0.85, 0.88]
    assert _gate(scores) is True
