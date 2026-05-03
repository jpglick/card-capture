import pytest
from card_capture.selector import CandidateSelector, ScoredCandidate
from card_capture.models import QualityScore

def test_candidate_selector_groups_by_time():
    candidates = [
        ScoredCandidate(detection_id=1, timestamp_ms=0, image_path="a.jpg", score=QualityScore(0.8, {})),
        ScoredCandidate(detection_id=2, timestamp_ms=100, image_path="b.jpg", score=QualityScore(0.9, {})),
        ScoredCandidate(detection_id=3, timestamp_ms=2000, image_path="c.jpg", score=QualityScore(0.7, {})),
    ]
    # frames_per_instance=1 to get only the best frame per group
    selected = CandidateSelector(group_gap_ms=1000, frames_per_instance=1).select(candidates)
    assert [c.detection_id for c in selected] == [2, 3]

def test_candidate_selector_keeps_multiple_per_instance():
    candidates = [
        ScoredCandidate(detection_id=1, timestamp_ms=0, image_path="a.jpg", score=QualityScore(0.8, {})),
        ScoredCandidate(detection_id=2, timestamp_ms=100, image_path="b.jpg", score=QualityScore(0.9, {})),
    ]
    selected = CandidateSelector(group_gap_ms=1000, frames_per_instance=2).select(candidates)
    assert len(selected) == 2
    assert [c.detection_id for c in selected] == [2, 1] # Sorted by score

def test_candidate_selector_respects_spatial_clustering():
    # Identical timestamps but different positions should split
    candidates = [
        ScoredCandidate(detection_id=1, timestamp_ms=0, image_path="a.jpg", score=QualityScore(0.8, {}), corners=((0,0),(10,0),(10,10),(0,10))),
        ScoredCandidate(detection_id=2, timestamp_ms=10, image_path="b.jpg", score=QualityScore(0.9, {}), corners=((100,100),(110,100),(110,110),(100,110))),
    ]
    # Spatial threshold 50, distance is ~141
    selected = CandidateSelector(spatial_variance_threshold=50, frames_per_instance=1).select(candidates)
    assert len(selected) == 2
