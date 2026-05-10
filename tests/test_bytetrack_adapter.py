import numpy as np

from card_capture.selector import ScoredCandidate
from card_capture.models import QualityScore
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
