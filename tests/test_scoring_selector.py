import numpy as np

from card_capture.models import CardDetection, CropResult, QualityScore
from card_capture.scoring import QualityScorer
from card_capture.selector import CandidateSelector, ScoredCandidate


def test_quality_scorer_ranks_sharp_image_above_blurry_image():
    sharp = np.zeros((120, 120, 3), dtype=np.uint8)
    sharp[::2] = 255
    blurry = np.full((120, 120, 3), 127, dtype=np.uint8)

    scorer = QualityScorer(target_pixels=120 * 120)

    sharp_score = scorer.score(sharp, detection_confidence=0.9)
    blurry_score = scorer.score(blurry, detection_confidence=0.9)

    assert sharp_score.total > blurry_score.total
    assert sharp_score.components["sharpness"] > blurry_score.components["sharpness"]


def test_quality_scorer_penalizes_overexposed_glare():
    normal = np.full((100, 100, 3), 120, dtype=np.uint8)
    overexposed = np.full((100, 100, 3), 255, dtype=np.uint8)

    scorer = QualityScorer(target_pixels=100 * 100)

    normal_score = scorer.score(normal, detection_confidence=0.9)
    overexposed_score = scorer.score(overexposed, detection_confidence=0.9)

    assert normal_score.total > overexposed_score.total
    assert normal_score.components["glare"] > overexposed_score.components["glare"]


def test_candidate_selector_keeps_best_candidate_per_time_group():
    candidates = [
        ScoredCandidate(detection_id=1, timestamp_ms=0, image_path="a.jpg", score=QualityScore(0.4, {})),
        ScoredCandidate(detection_id=2, timestamp_ms=400, image_path="b.jpg", score=QualityScore(0.9, {})),
        ScoredCandidate(detection_id=3, timestamp_ms=2500, image_path="c.jpg", score=QualityScore(0.7, {})),
    ]

    selected = CandidateSelector(group_gap_ms=1000, frames_per_instance=1).select(candidates)

    assert [candidate.detection_id for candidate in selected] == [2, 3]

def test_candidate_selector_respects_max_candidates_by_score():
    candidates = [
        ScoredCandidate(detection_id=1, timestamp_ms=0, image_path="a.jpg", score=QualityScore(0.4, {})),
        ScoredCandidate(detection_id=2, timestamp_ms=2000, image_path="b.jpg", score=QualityScore(0.9, {})),
        ScoredCandidate(detection_id=3, timestamp_ms=4000, image_path="c.jpg", score=QualityScore(0.7, {})),
    ]

    selected = CandidateSelector(group_gap_ms=1000, max_candidates=2).select(candidates)

    assert [candidate.detection_id for candidate in selected] == [2, 3, 1]

def test_quality_scorer_penalizes_wrong_aspect_ratio():
    """Portrait (≈0.714 w/h) should score higher on aspect_ratio than a square (mid-flip)."""
    # Portrait: w=63, h=88 → ratio ≈ 0.716 ≈ CARD_RATIO
    portrait = np.zeros((88, 63, 3), dtype=np.uint8)
    portrait[5:83, 5:58] = 120
    # Square: w=88, h=88 → ratio 1.0 (foreshortened)
    square = np.zeros((88, 88, 3), dtype=np.uint8)
    square[5:83, 5:83] = 120

    scorer = QualityScorer(target_pixels=88 * 88)
    portrait_score = scorer.score(portrait, detection_confidence=0.9)
    square_score = scorer.score(square, detection_confidence=0.9)

    assert "aspect_ratio" in portrait_score.components
    assert portrait_score.components["aspect_ratio"] > square_score.components["aspect_ratio"]


def test_quality_scorer_complexity_rewards_textured_image():
    """Textured image (card artwork) should score higher on complexity than uniform (plain back)."""
    rng = np.random.default_rng(42)
    textured = rng.integers(0, 256, (100, 70, 3), dtype=np.uint8)
    uniform = np.full((100, 70, 3), 128, dtype=np.uint8)

    scorer = QualityScorer(target_pixels=100 * 70)
    textured_score = scorer.score(textured, detection_confidence=0.9)
    uniform_score = scorer.score(uniform, detection_confidence=0.9)

    assert "complexity" in textured_score.components
    assert textured_score.components["complexity"] > uniform_score.components["complexity"]


def test_quality_scorer_has_six_components():
    """Score components dict must contain all six keys after the rebalance."""
    image = np.full((88, 63, 3), 128, dtype=np.uint8)
    scorer = QualityScorer(target_pixels=88 * 63)
    score = scorer.score(image, detection_confidence=1.0)

    assert set(score.components.keys()) == {
        "sharpness", "glare", "aspect_ratio", "size", "complexity", "confidence"
    }
