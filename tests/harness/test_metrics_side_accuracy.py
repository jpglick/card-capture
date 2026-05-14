"""Tests for harness.metrics.side_accuracy."""
import pytest
from pathlib import Path

from harness.metrics.side_accuracy import side_accuracy

FIX = Path("tests/harness/fixtures/runs")


def test_side_accuracy_one_correct_one_wrong():
    """Fixture: 2 matched, det_1 correct (Front), det_2 wrong (Back) → 0.5."""
    score = side_accuracy(
        db_path=FIX / "side_accuracy" / "cards.sqlite",
        truth_path=FIX / "side_accuracy" / "truth.json",
        video_id="side_accuracy",
    )
    assert score.value == pytest.approx(0.5)


def test_side_accuracy_all_correct():
    """all_matched: both detections are Front and GT expects Front → 1.0."""
    score = side_accuracy(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth_path=FIX / "all_matched" / "truth.json",
        video_id="all_matched",
    )
    assert score.value == pytest.approx(1.0)


def test_side_accuracy_undefined_when_no_matches(tmp_path):
    """No matched detections → side_accuracy is None."""
    truth_path = tmp_path / "t.json"
    truth_path.write_text(
        '{"video_id":"no_video","schema_version":1,"expected_cards":['
        '{"card_id":"c1","front_present":true,"back_present":false,'
        '"physical_card_key":"k1","is_foil":false}]}'
    )
    score = side_accuracy(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth_path=truth_path,
        video_id="no_video",
    )
    assert score.value is None
