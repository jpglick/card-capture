import numpy as np
import cv2

from card_capture.pipeline import _resolve_session_tracks, _PreparedTrack
from card_capture.selector import TrackState, ScoredCandidate
from card_capture.models import QualityScore
from card_capture.deduplicator import VisualDeduplicator


def _make_track(instance_id, image_paths):
    ts = TrackState(instance_id=instance_id)
    for i, p in enumerate(image_paths):
        ts.candidates.append(ScoredCandidate(
            detection_id=i, timestamp_ms=i * 33, image_path=str(p),
            score=QualityScore(total=0.7, components={}),
            corners=[(0, 0), (60, 0), (60, 90), (0, 90)],
            frame_index=i,
        ))
    return ts


def _save_card(tmp_path, name, fill, has_stripes=False):
    arr = np.zeros((200, 150, 3), dtype=np.uint8)
    arr[:] = fill
    # Add some texture so phash is non-degenerate
    arr[50:150, 30:120] = (fill[0] // 2, fill[1] // 2, fill[2] // 2)
    if has_stripes:
        # Add horizontal stripes to make the pattern very different
        for i in range(50, 150):
            if (i // 10) % 2 == 0:
                arr[i, 30:120] = np.uint8(np.clip(arr[i, 30:120] + 50, 0, 255))
    p = tmp_path / f"{name}.jpg"
    cv2.imwrite(str(p), arr)
    return p


def test_two_visually_similar_tracks_become_front_and_back(tmp_path):
    """Front+back of the SAME card (similar appearance) → second is labeled Back."""
    p_front = _save_card(tmp_path, "front", (200, 50, 50))
    p_back  = _save_card(tmp_path, "back",  (205, 55, 55))  # near-identical

    longer = _make_track("a", [p_front] * 5)
    shorter = _make_track("b", [p_back] * 3)
    prepared = [
        _PreparedTrack(
            track=longer, session_id=1, first_frame_index=0, angle="Front",
            frame_entries=[], canonical_entries=[], candidate_hashes=[],
            primary_hash="", side_score=0.0, appearance_vector=np.array([]),
            canonical_detection_ids=set(), best_canonical_detection_id=0,
            fused_canonical=None, duplicate_track_index=None
        ),
        _PreparedTrack(
            track=shorter, session_id=1, first_frame_index=0, angle="Front",
            frame_entries=[], canonical_entries=[], candidate_hashes=[],
            primary_hash="", side_score=0.0, appearance_vector=np.array([]),
            canonical_detection_ids=set(), best_canonical_detection_id=0,
            fused_canonical=None, duplicate_track_index=None
        ),
    ]
    _resolve_session_tracks(prepared, VisualDeduplicator())
    angles = {pt.track.instance_id: pt.angle for pt in prepared}
    assert angles == {"a": "Front", "b": "Back"}


def test_two_visually_distinct_tracks_remain_two_fronts(tmp_path):
    """Two DIFFERENT cards in one session must NOT be merged front+back."""
    p1 = _save_card(tmp_path, "card1", (220, 30, 30), has_stripes=False)
    p2 = _save_card(tmp_path, "card2", (30, 30, 220), has_stripes=True)  # very different

    t1 = _make_track("a", [p1] * 5)
    t2 = _make_track("b", [p2] * 3)
    prepared = [
        _PreparedTrack(
            track=t1, session_id=1, first_frame_index=0, angle="Front",
            frame_entries=[], canonical_entries=[], candidate_hashes=[],
            primary_hash="", side_score=0.0, appearance_vector=np.array([]),
            canonical_detection_ids=set(), best_canonical_detection_id=0,
            fused_canonical=None, duplicate_track_index=None
        ),
        _PreparedTrack(
            track=t2, session_id=1, first_frame_index=0, angle="Front",
            frame_entries=[], canonical_entries=[], candidate_hashes=[],
            primary_hash="", side_score=0.0, appearance_vector=np.array([]),
            canonical_detection_ids=set(), best_canonical_detection_id=0,
            fused_canonical=None, duplicate_track_index=None
        ),
    ]
    _resolve_session_tracks(prepared, VisualDeduplicator())
    angles = [pt.angle for pt in prepared]
    dups = [pt.duplicate_track_index for pt in prepared]
    assert angles == ["Front", "Front"], angles
    assert dups == [None, None], dups
