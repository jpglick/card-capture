import numpy as np

from card_capture.scoring import QualityScorer


def _card_like(h=900, w=600, has_occluder=False):
    """Synthesize a card-aspect crop. If has_occluder, paint a high-variance
    irregular blob into the bottom-right border strip."""
    arr = np.full((h, w, 3), 240, dtype=np.uint8)        # white border
    arr[40:h-40, 40:w-40] = 30                            # dark interior
    arr[60:h-60, 60:w-60] = np.random.randint(0, 200, (h - 120, w - 120, 3), dtype=np.uint8)
    if has_occluder:
        # Skin-toneish noisy blob intruding from bottom-right edge
        for y in range(h - 80, h):
            for x in range(w - 120, w):
                arr[y, x] = (210 + (x % 30), 160 + (y % 20), 130)
    return arr


def test_border_purity_penalizes_occluded_border():
    scorer = QualityScorer()
    clean = _card_like(has_occluder=False)
    dirty = _card_like(has_occluder=True)
    clean_score = scorer.score(clean, detection_confidence=0.8)
    dirty_score = scorer.score(dirty, detection_confidence=0.8)
    assert clean_score.total > dirty_score.total, (clean_score.total, dirty_score.total)
    assert "border_purity" in clean_score.components
    assert clean_score.components["border_purity"] > dirty_score.components["border_purity"]
