import json
from pathlib import Path

import pytest

from tests.regression.truth import ExpectedCard, GroundTruth, load_truth, TruthValidationError


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "video_001.truth.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_truth_returns_dataclass(tmp_path):
    path = _write(tmp_path, {
        "video_id": "video_001",
        "video_path": "tests/fixtures/golden_corpus/video_001/video_001.mp4",
        "labeled_at": "2026-05-10",
        "labeled_by": "josh",
        "expected_cards": [
            {
                "card_id": "card_001",
                "physical_card_key": "topps_chrome_2024_42",
                "front_present": True,
                "back_present": True,
                "approx_front_window_ms": [12500, 15800],
                "approx_back_window_ms": [16100, 18900],
                "notes": "foil",
            }
        ],
    })

    truth = load_truth(path)

    assert isinstance(truth, GroundTruth)
    assert truth.video_id == "video_001"
    assert len(truth.expected_cards) == 1
    card = truth.expected_cards[0]
    assert isinstance(card, ExpectedCard)
    assert card.card_id == "card_001"
    assert card.physical_card_key == "topps_chrome_2024_42"
    assert card.front_present is True
    assert card.approx_front_window_ms == (12500, 15800)


def test_load_truth_rejects_missing_video_id(tmp_path):
    path = _write(tmp_path, {"expected_cards": []})
    with pytest.raises(TruthValidationError, match="video_id"):
        load_truth(path)


def test_load_truth_allows_missing_optional_fields(tmp_path):
    path = _write(tmp_path, {
        "video_id": "video_002",
        "video_path": "x.mp4",
        "expected_cards": [
            {
                "card_id": "c1",
                "front_present": True,
                "back_present": False,
                "approx_front_window_ms": [0, 1000],
            }
        ],
    })
    truth = load_truth(path)
    card = truth.expected_cards[0]
    assert card.physical_card_key is None
    assert card.approx_back_window_ms is None
    assert card.notes == ""
