"""Tests for harness.schema — truth.json Pydantic models."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.schema import TruthFile, from_legacy

FIX = Path("tests/harness/fixtures/truth_examples")


def test_minimal_valid_parses():
    payload = json.loads((FIX / "minimal_valid.json").read_text())
    tf = TruthFile.model_validate(payload)
    assert tf.video_id == "v1"
    assert tf.schema_version == 1
    assert len(tf.expected_cards) == 1
    card = tf.expected_cards[0]
    assert card.card_id == "c1"
    assert card.front_present is True
    assert card.back_present is False
    assert card.physical_card_key == "k1"
    assert card.is_foil is False


def test_with_optional_fields_parses():
    payload = json.loads((FIX / "with_optional_fields.json").read_text())
    tf = TruthFile.model_validate(payload)
    card = tf.expected_cards[0]
    assert card.approx_front_window_ms == (1000, 2500)
    assert card.approx_back_window_ms == (3000, 4200)
    assert card.is_foil is True
    assert card.notes == "occluded by hand at 1.7s"


def test_missing_required_field_rejected():
    payload = json.loads((FIX / "missing_required.json").read_text())
    with pytest.raises(ValidationError):
        TruthFile.model_validate(payload)


def test_legacy_v0_converts():
    payload = json.loads((FIX / "legacy_v0.json").read_text())
    tf = from_legacy(payload)
    assert tf.schema_version == 1
    assert tf.video_id == "video_42"
    # card_001 had physical_card_key in the legacy file
    c1 = next(c for c in tf.expected_cards if c.card_id == "card_001")
    assert c1.physical_card_key == "charizard_base_4_holo_1999"
    assert c1.is_foil is False  # defaults to False for legacy
    # card_002 had no physical_card_key — should fall back to card_id
    c2 = next(c for c in tf.expected_cards if c.card_id == "card_002")
    assert c2.physical_card_key == "card_002"


def test_window_validation_rejects_inverted_range():
    payload = {
        "video_id": "v3",
        "schema_version": 1,
        "expected_cards": [
            {
                "card_id": "c1",
                "front_present": True,
                "back_present": False,
                "physical_card_key": "k1",
                "is_foil": False,
                "approx_front_window_ms": [5000, 2000],
            }
        ],
    }
    with pytest.raises(ValidationError):
        TruthFile.model_validate(payload)


def test_window_equal_start_end_allowed():
    """A zero-duration window [t, t] is allowed (start == end)."""
    payload = {
        "video_id": "v4",
        "schema_version": 1,
        "expected_cards": [
            {
                "card_id": "c1",
                "front_present": True,
                "back_present": False,
                "physical_card_key": "k1",
                "is_foil": False,
                "approx_front_window_ms": [3000, 3000],
            }
        ],
    }
    tf = TruthFile.model_validate(payload)
    assert tf.expected_cards[0].approx_front_window_ms == (3000, 3000)


def test_schema_version_zero_rejected():
    """schema_version must be >= 1; version 0 (legacy) must go through from_legacy()."""
    payload = {
        "video_id": "v5",
        "schema_version": 0,
        "expected_cards": [],
    }
    with pytest.raises(ValidationError):
        TruthFile.model_validate(payload)


def test_from_legacy_already_v1_passthrough():
    """from_legacy passes through a v1 file unchanged."""
    payload = json.loads((FIX / "minimal_valid.json").read_text())
    tf = from_legacy(payload)
    assert tf.schema_version == 1
    assert tf.video_id == "v1"


def test_model_is_frozen():
    """TruthFile and ExpectedCard are immutable (frozen=True)."""
    payload = json.loads((FIX / "minimal_valid.json").read_text())
    tf = TruthFile.model_validate(payload)
    with pytest.raises(Exception):
        tf.video_id = "mutated"  # type: ignore[misc]
