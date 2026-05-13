"""Tests for harness.metrics.card_precision."""
import pytest
from pathlib import Path

from harness.metrics.card_precision import card_precision

FIX = Path("tests/harness/fixtures/runs")


def test_precision_2_of_3():
    """Fixture has 3 detections (2 matched, 1 phantom) → precision = 2/3."""
    score = card_precision(
        db_path=FIX / "one_phantom" / "cards.sqlite",
        truth_path=FIX / "one_phantom" / "truth.json",
        video_id="one_phantom",
    )
    assert score == pytest.approx(2 / 3)


def test_precision_undefined_when_no_detections(tmp_path):
    """0 detections in DB (unknown video_id) → precision is undefined (None)."""
    truth_path = tmp_path / "t.json"
    truth_path.write_text(
        '{"video_id":"no_video","schema_version":1,"expected_cards":['
        '{"card_id":"c1","front_present":true,"back_present":false,'
        '"physical_card_key":"k1","is_foil":false}]}'
    )
    score = card_precision(
        db_path=FIX / "one_phantom" / "cards.sqlite",
        truth_path=truth_path,
        video_id="no_video",
    )
    assert score is None


def test_precision_zero_when_no_gt_cards(tmp_path):
    """GT is empty → all detections are phantoms → precision = 0."""
    truth_path = tmp_path / "empty.json"
    truth_path.write_text(
        '{"video_id":"one_phantom","schema_version":1,"expected_cards":[]}'
    )
    score = card_precision(
        db_path=FIX / "one_phantom" / "cards.sqlite",
        truth_path=truth_path,
        video_id="one_phantom",
    )
    assert score == pytest.approx(0.0)


def test_precision_perfect_when_all_detections_matched():
    """all_matched fixture: 2 detections, both matched → precision = 1.0."""
    score = card_precision(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth_path=FIX / "all_matched" / "truth.json",
        video_id="all_matched",
    )
    assert score == pytest.approx(1.0)
