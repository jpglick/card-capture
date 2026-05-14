"""Tests for FBPredictor refusing predictions without a checkpoint.

Closes V4_CONCERNS §1.5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from card_capture.ml.errors import UntrainedModelError
from card_capture.ml.inference.fb_predict import FBPredictor


def test_predictor_refuses_when_no_checkpoint_path():
    with pytest.raises(UntrainedModelError):
        FBPredictor(checkpoint_path=None)


def test_predictor_refuses_when_checkpoint_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(UntrainedModelError):
        FBPredictor(checkpoint_path=missing)


def test_is_available_returns_false_when_no_checkpoint(tmp_path: Path):
    assert FBPredictor.is_available(None) is False
    assert FBPredictor.is_available(tmp_path / "missing.pt") is False


def test_is_available_returns_true_when_checkpoint_exists(tmp_path: Path):
    # Write any non-empty file at the path; we're testing the presence
    # check, not the load.
    fake_ckpt = tmp_path / "fake.pt"
    fake_ckpt.write_bytes(b"x")
    assert FBPredictor.is_available(fake_ckpt) is True


def test_predictor_loads_and_predicts_with_real_checkpoint(tmp_path: Path):
    """When given a real checkpoint, the predictor loads and returns
    a (label, confidence) tuple."""
    import torch
    from card_capture.ml.fb_classifier import FBClassifier

    # Save a freshly-constructed model as a checkpoint — its
    # predictions will be near-random, but the API contract must hold.
    ckpt = tmp_path / "fb.pt"
    torch.save(FBClassifier(pretrained=False).state_dict(), ckpt)

    pred = FBPredictor(checkpoint_path=ckpt)
    img = np.zeros((1050, 750, 3), dtype=np.uint8)  # arbitrary BGR
    label, conf = pred.predict(img)

    assert label in ("front", "back")
    assert 0.0 <= conf <= 1.0
