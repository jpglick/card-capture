"""Tests for harness.match — detection↔truth matcher."""
from pathlib import Path

import pytest

from harness.match import MatchPair, match_detections_to_truth
from harness.schema import TruthFile

FIX = Path("tests/harness/fixtures/runs")


# ---------------------------------------------------------------------------
# all_matched fixture: 3 GT cards, 2 detections (card_03 is missed)
# ---------------------------------------------------------------------------


def test_all_matched_returns_correct_pair_count():
    truth = TruthFile.model_validate_json(
        (FIX / "all_matched" / "truth.json").read_text()
    )
    pairs = match_detections_to_truth(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth=truth,
        video_id="all_matched",
    )
    # 3 GT cards → 3 slots; 2 have detections, 1 is missed.
    assert len(pairs) == 3


def test_all_matched_returns_correct_pairs():
    truth = TruthFile.model_validate_json(
        (FIX / "all_matched" / "truth.json").read_text()
    )
    pairs = match_detections_to_truth(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth=truth,
        video_id="all_matched",
    )
    # 2 matches (both gt_card_id and detection_id present)
    assert sum(1 for p in pairs if p.gt_card_id and p.detection_id) == 2
    # 1 unmatched GT card (gt_card_id set, no detection_id)
    assert sum(1 for p in pairs if p.gt_card_id and not p.detection_id) == 1
    # 0 phantoms (all detections were matched)
    assert sum(1 for p in pairs if not p.gt_card_id and p.detection_id) == 0


def test_all_matched_side_match_is_correct():
    """Both detections are Front, GT expects Front → side_match=True."""
    truth = TruthFile.model_validate_json(
        (FIX / "all_matched" / "truth.json").read_text()
    )
    pairs = match_detections_to_truth(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth=truth,
        video_id="all_matched",
    )
    matched_pairs = [p for p in pairs if p.gt_card_id and p.detection_id]
    assert all(p.side_match for p in matched_pairs)


# ---------------------------------------------------------------------------
# one_phantom fixture: 2 GT cards, 3 detections (det_3 is phantom)
# ---------------------------------------------------------------------------


def test_one_phantom_detects_phantom():
    truth = TruthFile.model_validate_json(
        (FIX / "one_phantom" / "truth.json").read_text()
    )
    pairs = match_detections_to_truth(
        db_path=FIX / "one_phantom" / "cards.sqlite",
        truth=truth,
        video_id="one_phantom",
    )
    assert sum(1 for p in pairs if not p.gt_card_id and p.detection_id) == 1


# ---------------------------------------------------------------------------
# side_accuracy fixture: det_1 correct (Front), det_2 wrong (Back vs Front)
# ---------------------------------------------------------------------------


def test_side_accuracy_fixture_side_match_values():
    truth = TruthFile.model_validate_json(
        (FIX / "side_accuracy" / "truth.json").read_text()
    )
    pairs = match_detections_to_truth(
        db_path=FIX / "side_accuracy" / "cards.sqlite",
        truth=truth,
        video_id="side_accuracy",
    )
    matched = [p for p in pairs if p.gt_card_id and p.detection_id]
    assert len(matched) == 2
    correct = sum(1 for p in matched if p.side_match)
    assert correct == 1


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


def test_empty_truth_returns_no_gt_pairs(tmp_path):
    truth_path = tmp_path / "empty.json"
    truth_path.write_text(
        '{"video_id":"x","schema_version":1,"expected_cards":[]}'
    )
    truth = TruthFile.model_validate_json(truth_path.read_text())
    pairs = match_detections_to_truth(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth=truth,
        video_id="all_matched",
    )
    # No GT slots → all detections are phantoms
    assert all(p.gt_card_id is None for p in pairs)


def test_unknown_video_id_returns_all_unmatched(tmp_path):
    """A video_id that doesn't exist in the DB → no detections → all GT missed."""
    truth_path = tmp_path / "t.json"
    truth_path.write_text(
        '{"video_id":"nonexistent","schema_version":1,"expected_cards":['
        '{"card_id":"c1","front_present":true,"back_present":false,'
        '"physical_card_key":"k1","is_foil":false}]}'
    )
    truth = TruthFile.model_validate_json(truth_path.read_text())
    pairs = match_detections_to_truth(
        db_path=FIX / "all_matched" / "cards.sqlite",
        truth=truth,
        video_id="nonexistent",
    )
    assert len(pairs) == 1
    assert pairs[0].gt_card_id == "c1"
    assert pairs[0].detection_id is None
