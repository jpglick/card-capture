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

    selected = CandidateSelector(group_gap_ms=1000, max_candidates=10).select(candidates)

    assert [candidate.detection_id for candidate in selected] == [2, 3]


def test_candidate_selector_respects_max_candidates_by_score():
    candidates = [
        ScoredCandidate(detection_id=1, timestamp_ms=0, image_path="a.jpg", score=QualityScore(0.4, {})),
        ScoredCandidate(detection_id=2, timestamp_ms=2000, image_path="b.jpg", score=QualityScore(0.9, {})),
        ScoredCandidate(detection_id=3, timestamp_ms=4000, image_path="c.jpg", score=QualityScore(0.7, {})),
    ]

    selected = CandidateSelector(group_gap_ms=1000, max_candidates=2).select(candidates)

    assert [candidate.detection_id for candidate in selected] == [2, 3]
