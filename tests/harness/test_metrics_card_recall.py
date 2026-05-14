"""Tests for harness.metrics.card_recall."""
import pytest
from pathlib import Path

from harness.metrics.card_recall import card_recall

FIX = Path("tests/harness/fixtures/runs")


def test_recall_2_of_3():
    """Fixture has 3 GT cards; 2 matched, 1 missed → recall = 2/3."""
    score = card_recall(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth_path=FIX / "all_matched" / "truth.json",
        video_id="all_matched",
    )
    assert score.value == pytest.approx(2 / 3)


def test_recall_undefined_when_no_truth(tmp_path):
    """0 GT cards → recall is undefined (None)."""
    truth_path = tmp_path / "empty.json"
    truth_path.write_text(
        '{"video_id":"x","schema_version":1,"expected_cards":[]}'
    )
    score = card_recall(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth_path=truth_path,
        video_id="x",
    )
    assert score.value is None


def test_recall_zero_when_no_detections(tmp_path):
    """GT cards present but DB has no detections for this video_id → recall = 0."""
    truth_path = tmp_path / "t.json"
    truth_path.write_text(
        '{"video_id":"no_video","schema_version":1,"expected_cards":['
        '{"card_id":"c1","front_present":true,"back_present":false,'
        '"physical_card_key":"k1","is_foil":false}]}'
    )
    score = card_recall(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth_path=truth_path,
        video_id="no_video",
    )
    assert score.value == 0.0
