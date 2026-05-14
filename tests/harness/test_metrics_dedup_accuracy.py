"""Tests for harness.metrics.dedup_accuracy."""
import pytest
from pathlib import Path

from harness.metrics.dedup_accuracy import dedup_accuracy
from harness.metrics.types import MetricResult

FIX = Path("tests/harness/fixtures/runs")


def test_dedup_perfect_grouping():
    """Dedup fixture: predicted clusters match GT clusters exactly → ARI = 1.0."""
    result = dedup_accuracy(
        db_path=FIX / "dedup" / "cards.sqlite",
        truth_path=FIX / "dedup" / "truth.json",
        video_id="dedup",
    )
    assert isinstance(result, MetricResult)
    assert result.breakdown["ari"] == pytest.approx(1.0, abs=1e-6)
    assert result.breakdown["pair_f1"] == pytest.approx(1.0, abs=1e-6)


def test_dedup_returns_none_when_fewer_than_two_matches(tmp_path):
    """Fewer than 2 matched detections → ARI and pair_f1 are both None."""
    truth_path = tmp_path / "t.json"
    truth_path.write_text(
        '{"video_id":"dedup","schema_version":1,"expected_cards":['
        '{"card_id":"card_01","front_present":true,"back_present":false,'
        '"physical_card_key":"key_pair","is_foil":false,'
        '"approx_front_window_ms":[1000,2000]}]}'
    )
    result = dedup_accuracy(
        db_path=FIX / "dedup" / "cards.sqlite",
        truth_path=truth_path,
        video_id="dedup",
    )
    assert result.breakdown["ari"] is None
    assert result.breakdown["pair_f1"] is None


def test_dedup_result_is_frozen():
    """MetricResult instances are immutable."""
    d = MetricResult(breakdown={"ari": 0.5, "pair_f1": 0.5})
    with pytest.raises(Exception):
        d.breakdown = {}  # type: ignore[misc]
