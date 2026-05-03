import pytest
from dataclasses import field
from typing import Optional, List, Tuple
from card_capture.selector import ScoredCandidate, QualityScore
from card_capture.selector import HysteresisTracker, TrackState

def test_hysteresis_tracker_starts_new_track_on_high_score():
    tracker = HysteresisTracker(t_high=0.55, t_low=0.20, max_dist=75.0)
    
    # Candidate with score > t_high should start a new track
    candidate = ScoredCandidate(
        detection_id=1,
        timestamp_ms=100,
        image_path="test.jpg",
        score=QualityScore(0.6, {}),
        corners=[(0, 0), (10, 0), (10, 10), (0, 10)]
    )
    
    tracker.process(candidate)
    
    assert len(tracker.active_tracks) == 1
    track = tracker.active_tracks[0]
    assert track.candidates == [candidate]
    assert track.last_centroid == (5.0, 5.0)
    assert track.active is True

def test_hysteresis_tracker_ignores_low_score_initially():
    tracker = HysteresisTracker(t_high=0.55, t_low=0.20, max_dist=75.0)
    
    # Candidate with score < t_high should NOT start a new track if no tracks exist
    candidate = ScoredCandidate(
        detection_id=1,
        timestamp_ms=100,
        image_path="test.jpg",
        score=QualityScore(0.4, {}),
        corners=[(0, 0), (10, 0), (10, 10), (0, 10)]
    )
    
    tracker.process(candidate)
    
    assert len(tracker.active_tracks) == 0

def test_hysteresis_tracker_maintains_track_on_medium_score():
    tracker = HysteresisTracker(t_high=0.55, t_low=0.20, max_dist=75.0)
    
    # 1. Start track
    c1 = ScoredCandidate(
        detection_id=1,
        timestamp_ms=100,
        image_path="test1.jpg",
        score=QualityScore(0.6, {}),
        corners=[(0, 0), (10, 0), (10, 10), (0, 10)]
    )
    tracker.process(c1)
    
    # 2. Add candidate with t_low < score < t_high and small distance
    c2 = ScoredCandidate(
        detection_id=2,
        timestamp_ms=200,
        image_path="test2.jpg",
        score=QualityScore(0.3, {}),
        corners=[(1, 1), (11, 1), (11, 11), (1, 11)]
    )
    tracker.process(c2)
    
    assert len(tracker.active_tracks) == 1
    track = tracker.active_tracks[0]
    assert track.candidates == [c1, c2]
    assert track.last_centroid == (6.0, 6.0)

def test_hysteresis_tracker_starts_new_track_on_large_distance():
    tracker = HysteresisTracker(t_high=0.55, t_low=0.20, max_dist=75.0)
    
    # 1. Start track
    c1 = ScoredCandidate(
        detection_id=1,
        timestamp_ms=100,
        image_path="test1.jpg",
        score=QualityScore(0.6, {}),
        corners=[(0, 0), (10, 0), (10, 10), (0, 10)]
    )
    tracker.process(c1)
    
    # 2. Candidate with high score but far away should start its OWN track
    # (In this simple version, if it's far from ALL active tracks, it starts a new one)
    c2 = ScoredCandidate(
        detection_id=2,
        timestamp_ms=200,
        image_path="test2.jpg",
        score=QualityScore(0.7, {}),
        corners=[(200, 200), (210, 200), (210, 210), (200, 210)]
    )
    tracker.process(c2)
    
    assert len(tracker.active_tracks) == 2
    assert tracker.active_tracks[0].candidates == [c1]
    assert tracker.active_tracks[1].candidates == [c2]

def test_hysteresis_tracker_ignores_medium_score_with_no_nearby_track():
    tracker = HysteresisTracker(t_high=0.55, t_low=0.20, max_dist=75.0)
    
    # Candidate with t_low < score < t_high but NO nearby track should be ignored
    candidate = ScoredCandidate(
        detection_id=1,
        timestamp_ms=100,
        image_path="test.jpg",
        score=QualityScore(0.4, {}),
        corners=[(0, 0), (10, 0), (10, 10), (0, 10)]
    )
    
    tracker.process(candidate)
    
    assert len(tracker.active_tracks) == 0
