import numpy as np

from card_capture.core.models import QualityScore, ScoredCandidate
from card_capture.tracking import ByteTrackAdapter


def _candidate(detection_id, frame_index, x, y, conf=0.9, w=200, h=300):
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return ScoredCandidate(
        detection_id=detection_id,
        timestamp_ms=frame_index * 33,
        image_path="x.jpg",
        score=QualityScore(total=conf, components={"sharpness": conf, "blur": 0.0, "area": 0.5}),
        corners=corners,
        frame_index=frame_index,
    )


def test_adapter_assigns_consistent_track_id_for_overlapping_boxes():
    adapter = ByteTrackAdapter()
    out = []
    for i in range(5):
        cand = _candidate(detection_id=i, frame_index=i, x=100, y=100)
        out.append(adapter.process([cand]))

    track_ids = [a[0].track_id for a in out if a]
    assert len(set(track_ids)) == 1, f"expected single track, got {track_ids}"


def test_adapter_starts_new_track_for_distant_box():
    adapter = ByteTrackAdapter()
    # Process two distant boxes simultaneously
    a = _candidate(0, 0, x=100, y=100)
    b = _candidate(1, 0, x=1500, y=1500)  # very far away
    out = adapter.process([a, b])
    assert len(out) == 2, f"Expected 2 detections, got {len(out)}"
    # They should get different track IDs
    track_ids = [d.track_id for d in out]
    assert len(set(track_ids)) == 2, f"Expected 2 different tracks, got {track_ids}"


def test_adapter_finalize_returns_track_states():
    adapter = ByteTrackAdapter(min_track_length=2)
    for i in range(3):
        adapter.process([_candidate(i, i, x=100, y=100)])
    tracks = adapter.finalize()
    assert len(tracks) >= 1
    track = tracks[0]
    assert hasattr(track, "candidates")
    assert hasattr(track, "instance_id")
    assert len(track.candidates) >= 2


def test_adapter_skips_candidates_without_corners():
    """Regression: index mismatch when some candidates lack corners.
    
    ByteTrack only tracks detections that pass confidence threshold.
    When some detections are skipped:
    - valid_candidates is smaller than raw input
    - ByteTrack.update_with_detections() sets tracker_id in-place on det (which is valid_candidates size)
    - We iterate det.tracker_id to match original indices correctly.
    
    This test verifies the fix: don't iterate over the filtered output; 
    iterate over the original-indexed det.tracker_id instead.
    """
    adapter = ByteTrackAdapter()
    cand_no_corners = ScoredCandidate(
        detection_id=99,
        timestamp_ms=0,
        image_path="x.jpg",
        score=QualityScore(total=0.9, components={"sharpness": 0.9, "blur": 0.0, "area": 0.5}),
        corners=None,
        frame_index=0,
    )
    cand_with_corners = _candidate(detection_id=1, frame_index=0, x=100, y=100)
    # Should not crash; candidate with corners should be processed
    out = adapter.process([cand_no_corners, cand_with_corners])
    # out may be empty (ByteTrack needs a few frames to confirm a track) but should not crash
    assert isinstance(out, list)


def test_adapter_first_frame_does_not_crash_when_tracker_id_is_none():
    """Regression: update_with_detections mutates det.tracker_id in-place;
    on the very first call tracker_id may remain None if ByteTrack hasn't
    confirmed any tracks yet. process() must guard against this."""
    adapter = ByteTrackAdapter()
    cand = _candidate(detection_id=0, frame_index=0, x=100, y=100)
    # Single call on a fresh adapter — ByteTrack hasn't seen enough frames to
    # confirm a track, so tracker_id can be None. Must return a list, not raise.
    result = adapter.process([cand])
    assert isinstance(result, list)


def test_adapter_returns_empty_not_none_on_first_frame():
    """process() must always return a list, even when no tracks are confirmed."""
    adapter = ByteTrackAdapter()
    result = adapter.process([_candidate(0, 0, x=200, y=200)])
    assert result is not None
    assert isinstance(result, list)
