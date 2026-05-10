import os
from pathlib import Path

import numpy as np
import pytest

from card_capture.presence.classifier import PresenceClassifier

WEIGHTS = Path("models/presence_classifier.pt")
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="weights not trained yet")


def test_classifier_returns_score_in_unit_interval():
    clf = PresenceClassifier(weights_path=WEIGHTS)
    frame = np.full((300, 300, 3), 200, dtype=np.uint8)
    score = clf.score(frame)
    assert 0.0 <= score <= 1.0


def test_classifier_batch_returns_list():
    clf = PresenceClassifier(weights_path=WEIGHTS)
    frames = [np.full((300, 300, 3), v, dtype=np.uint8) for v in (50, 100, 150, 200)]
    scores = clf.score_batch(frames)
    assert len(scores) == 4
    for s in scores:
        assert 0.0 <= s <= 1.0
