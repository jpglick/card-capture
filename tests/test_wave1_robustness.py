"""
Integration tests for Wave 1 robustness improvements.

Task 1: Real frame passing to BoT-SORT for ReID restoration.
- Verify that BoT-SORT receives real frame data (not zeros)
- Verify that OSNet embeddings are computed (not all zeros)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from card_capture.models import (
    CornerDetection,
    DetectionPacket,
    FrameSample,
    QualityScore,
)
from card_capture.selector import ScoredCandidate
from card_capture.tracking.botsort_adapter import BoTSORTAdapter


def _candidate_with_frame_path(
    detection_id: int,
    frame_index: int,
    source_frame_path: str,
    x: float = 100.0,
    y: float = 100.0,
    conf: float = 0.9,
    w: float = 200.0,
    h: float = 300.0,
) -> ScoredCandidate:
    """Create a ScoredCandidate with a real frame path."""
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return ScoredCandidate(
        detection_id=detection_id,
        timestamp_ms=frame_index * 33,
        image_path=source_frame_path,
        score=QualityScore(
            total=conf,
            components={"sharpness": conf, "blur": 0.0, "area": 0.5}
        ),
        corners=corners,
        frame_index=frame_index,
    )


@pytest.fixture
def mock_botsort_adapter():
    """BoTSORTAdapter with boxmot mocked to capture frames passed to update()."""

    class MockDetections:
        def __init__(self, xyxy, confidence, class_id):
            self.xyxy = xyxy
            self.confidence = confidence
            self.class_id = class_id
            self.tracker_id = None

    captured_frames = []

    def mock_update(det, img):
        """Capture the frame passed to update()."""
        captured_frames.append(img)
        # Simulate normal track assignment
        n = len(det.xyxy)
        det.tracker_id = np.ones(n, dtype=int)

    mock_supervision = MagicMock()
    mock_supervision.Detections = MockDetections

    with patch.dict("sys.modules", {
        "boxmot": MagicMock(),
        "boxmot.trackers": MagicMock(),
        "boxmot.trackers.botsort": MagicMock(),
        "boxmot.trackers.botsort.botsort": MagicMock(),
        "supervision": mock_supervision,
    }):
        with patch("card_capture.tracking.botsort_adapter._import_botsort") as mock_import:
            MockBoTSORT = MagicMock()
            mock_tracker = MagicMock()
            mock_tracker.update = mock_update
            MockBoTSORT.return_value = mock_tracker
            mock_import.return_value = MockBoTSORT

            from card_capture.tracking.botsort_adapter import BoTSORTAdapter

            adapter = BoTSORTAdapter(min_track_length=1)
            adapter._captured_frames = captured_frames  # Expose for testing
            yield adapter


def test_botsort_receives_real_frames_not_zeros(mock_botsort_adapter):
    """
    Test that BoT-SORT receives real frame data when frame_path is provided.

    This verifies the core requirement of Task 1: instead of passing np.zeros
    to the tracker.update() call, pass the actual decoded frame from disk.

    Currently this test FAILS because the adapter still passes dummy_img.
    After implementation, it should PASS.
    """
    # Create a temporary directory with sample frame images
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create two distinct test frames
        # Frame 0: red card (high red channel)
        frame0_path = tmpdir / "frame_0.jpg"
        frame0 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame0[100:400, 150:550, 2] = 200  # Red channel
        import cv2
        cv2.imwrite(str(frame0_path), frame0)

        # Frame 1: blue card (high blue channel)
        frame1_path = tmpdir / "frame_1.jpg"
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[100:400, 150:550, 0] = 200  # Blue channel
        cv2.imwrite(str(frame1_path), frame1)

        # Create candidates pointing to real frames
        cand0 = _candidate_with_frame_path(
            detection_id=0,
            frame_index=0,
            source_frame_path=str(frame0_path),
            x=150.0,
            y=100.0,
        )
        cand1 = _candidate_with_frame_path(
            detection_id=1,
            frame_index=1,
            source_frame_path=str(frame1_path),
            x=150.0,
            y=100.0,
        )

        # Process both candidates
        mock_botsort_adapter.process([cand0])
        mock_botsort_adapter.process([cand1])

        # Verify frames were captured
        assert len(mock_botsort_adapter._captured_frames) == 2, \
            f"Expected 2 frames captured, got {len(mock_botsort_adapter._captured_frames)}"

        frame0_received = mock_botsort_adapter._captured_frames[0]
        frame1_received = mock_botsort_adapter._captured_frames[1]

        # CRITICAL ASSERTION: frames should NOT be all zeros
        # (currently fails because adapter uses dummy_img)
        assert frame0_received is not None, "Frame 0 should not be None"
        assert frame1_received is not None, "Frame 1 should not be None"

        # Verify frames are actually different (red vs blue)
        # After implementation, these should be the real frames from disk
        assert frame0_received.max() > 0, "Frame 0 should contain non-zero data (real frame)"
        assert frame1_received.max() > 0, "Frame 1 should contain non-zero data (real frame)"

        # Verify the frames have different channel distributions
        # (red card has high red channel, blue card has high blue channel)
        frame0_red_mean = float(frame0_received[:, :, 2].mean())
        frame0_blue_mean = float(frame0_received[:, :, 0].mean())
        frame1_red_mean = float(frame1_received[:, :, 2].mean())
        frame1_blue_mean = float(frame1_received[:, :, 0].mean())

        # Frame 0 should have more red than blue
        assert frame0_red_mean > frame0_blue_mean, \
            f"Frame 0 should be red-dominant: red={frame0_red_mean:.1f}, blue={frame0_blue_mean:.1f}"

        # Frame 1 should have more blue than red
        assert frame1_blue_mean > frame1_red_mean, \
            f"Frame 1 should be blue-dominant: blue={frame1_blue_mean:.1f}, red={frame1_red_mean:.1f}"


def test_botsort_fallback_to_zeros_on_decode_failure(mock_botsort_adapter):
    """
    Test that BoT-SORT falls back to zeros if frame decode fails.

    This verifies graceful degradation: if frame_path is invalid or decode
    fails, the adapter should fall back to the current np.zeros behavior
    rather than crashing.

    Currently this test may FAIL because fallback logic is not yet implemented.
    After implementation, it should PASS.
    """
    # Create a candidate with a non-existent frame path
    cand = _candidate_with_frame_path(
        detection_id=0,
        frame_index=0,
        source_frame_path="/nonexistent/path/to/frame.jpg",
    )

    # Should not raise an exception
    result = mock_botsort_adapter.process([cand])

    # Should still return adapted detections
    assert isinstance(result, list)


def test_front_back_assignment_uses_side_score(tmp_path):
    """
    Task 2: Front/Back assignment should use side_score (textiness) as primary sort key.

    High textiness (0.8+) → Front (image-rich side)
    Low textiness (0.2-) → Back (uniform color side)

    This test creates two mock tracks with different side_score values and verifies
    that the high-textiness track is selected as Front regardless of track length.
    """
    import cv2
    from card_capture.pipeline import _resolve_session_tracks, _PreparedTrack
    from card_capture.selector import TrackState, ScoredCandidate
    from card_capture.models import QualityScore
    from card_capture.deduplicator import VisualDeduplicator

    # Helper to create a track with specified side_score
    def make_track_with_score(instance_id, num_frames, side_score_val):
        ts = TrackState(instance_id=instance_id)
        # Create dummy frame images
        for i in range(num_frames):
            frame_path = tmp_path / f"{instance_id}_frame_{i}.jpg"
            frame = np.zeros((200, 150, 3), dtype=np.uint8)
            frame[:] = (100, 100, 100)
            cv2.imwrite(str(frame_path), frame)

            ts.candidates.append(ScoredCandidate(
                detection_id=i,
                timestamp_ms=i * 33,
                image_path=str(frame_path),
                score=QualityScore(total=0.7, components={}),
                corners=[(0, 0), (60, 0), (60, 90), (0, 90)],
                frame_index=i,
            ))
        return ts, side_score_val

    # Create high-textiness track (Front candidate)
    track_high, score_high = make_track_with_score("high_text", num_frames=3, side_score_val=0.8)

    # Create low-textiness track (Back candidate) - NOTE: MORE frames but lower textiness
    track_low, score_low = make_track_with_score("low_text", num_frames=5, side_score_val=0.2)

    # Build PreparedTrack objects with explicit side_score values
    prepared = [
        _PreparedTrack(
            track=track_high,
            session_id=1,
            first_frame_index=0,
            angle="Front",
            frame_entries=[],
            canonical_entries=[],
            candidate_hashes=[],
            primary_hash="",
            side_score=score_high,  # HIGH textiness
            appearance_vector=np.array([]),
            canonical_detection_ids=set(),
            duplicate_track_index=None
        ),
        _PreparedTrack(
            track=track_low,
            session_id=1,
            first_frame_index=0,
            angle="Front",
            frame_entries=[],
            canonical_entries=[],
            candidate_hashes=[],
            primary_hash="",
            side_score=score_low,  # LOW textiness (but more frames)
            appearance_vector=np.array([]),
            canonical_detection_ids=set(),
            duplicate_track_index=None
        ),
    ]

    # Before refactoring, this test will fail because length-based sort gives
    # the 5-frame track priority. After refactoring to use side_score, high-textiness
    # should be Front.
    _resolve_session_tracks(prepared, VisualDeduplicator())

    angles = {pt.track.instance_id: pt.angle for pt in prepared}
    # High textiness should be Front, low should be Back (if same card detection passes)
    assert angles["high_text"] == "Front", \
        f"High-textiness track should be Front, got {angles['high_text']}"
    assert angles["low_text"] == "Back", \
        f"Low-textiness track should be Back, got {angles['low_text']}"
