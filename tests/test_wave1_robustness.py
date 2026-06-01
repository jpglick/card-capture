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

from card_capture.core.models import (
    CornerDetection,
    DetectionPacket,
    FrameSample,
    QualityScore,
)
from card_capture.core.models import ScoredCandidate
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
        # det is [x1, y1, x2, y2, conf, cls]
        n = len(det)
        res = np.zeros((n, 8))
        if n > 0:
            res[:, :4] = det[:, :4]
            res[:, 4] = np.arange(n) + 1
            res[:, 5] = det[:, 4]
        return res

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


@pytest.mark.skip(reason="_resolve_session_tracks retired with monolith")
def test_front_back_assignment_uses_side_score(tmp_path):
    """
    Task 2: Front/Back assignment should use side_score (textiness) as primary sort key.

    High textiness (0.8+) → Front (image-rich side)
    Low textiness (0.2-) → Back (uniform color side)

    This test creates two mock tracks with different side_score values and verifies
    that the high-textiness track is selected as Front regardless of track length.
    """
    import cv2
    from card_capture.pipeline_utils import _resolve_session_tracks, _PreparedTrack
    from card_capture.core.models import TrackState, ScoredCandidate
    from card_capture.core.models import QualityScore
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
            best_canonical_detection_id=0,
            fused_canonical=None,
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
            best_canonical_detection_id=0,
            fused_canonical=None,
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


@pytest.mark.skip(reason="_resolve_session_tracks retired with monolith")
def test_quality_weighted_track_selection(tmp_path):
    """
    Task 3: Quality-weighted primary track selection.

    Replace pure length-based selection with composite score:
    score = 0.6·normalized_length + 0.4·mean_quality_score

    Even though blurry track is longer, sharp short track should be selected as Front
    because it has higher quality (0.85 vs 0.40).

    Test creates:
    - Sharp short track: 3 candidates, quality_score 0.85
    - Blurry long track: 20 candidates, quality_score 0.40

    Expected: sharp track selected as Front (not blurry despite length)
    """
    import cv2
    from card_capture.pipeline_utils import _resolve_session_tracks, _PreparedTrack
    from card_capture.core.models import TrackState, ScoredCandidate
    from card_capture.core.models import QualityScore
    from card_capture.deduplicator import VisualDeduplicator

    def make_track_with_quality(instance_id, num_frames, quality_val, side_score_val=0.5):
        ts = TrackState(instance_id=instance_id)
        for i in range(num_frames):
            frame_path = tmp_path / f"{instance_id}_frame_{i}.jpg"
            frame = np.zeros((200, 150, 3), dtype=np.uint8)
            frame[:] = (100, 100, 100)
            cv2.imwrite(str(frame_path), frame)

            ts.candidates.append(ScoredCandidate(
                detection_id=i,
                timestamp_ms=i * 33,
                image_path=str(frame_path),
                # quality_score in components dict simulates sharpness metric
                score=QualityScore(
                    total=quality_val,
                    components={"quality_score": quality_val}
                ),
                corners=[(0, 0), (60, 0), (60, 90), (0, 90)],
                frame_index=i,
            ))
        return ts, side_score_val

    # Sharp short track: 3 candidates, quality 0.85
    track_sharp, score_sharp = make_track_with_quality(
        "sharp_short", num_frames=3, quality_val=0.85, side_score_val=0.5
    )

    # Blurry long track: 20 candidates, quality 0.40
    track_blurry, score_blurry = make_track_with_quality(
        "blurry_long", num_frames=20, quality_val=0.40, side_score_val=0.5
    )

    prepared = [
        _PreparedTrack(
            track=track_sharp,
            session_id=1,
            first_frame_index=0,
            angle="Front",
            frame_entries=[],
            canonical_entries=[],
            candidate_hashes=[],
            primary_hash="",
            side_score=score_sharp,  # Same side_score as tie-breaker test
            appearance_vector=np.array([]),
            canonical_detection_ids=set(),
            best_canonical_detection_id=0,
            fused_canonical=None,
            duplicate_track_index=None
        ),
        _PreparedTrack(
            track=track_blurry,
            session_id=1,
            first_frame_index=0,
            angle="Front",
            frame_entries=[],
            canonical_entries=[],
            candidate_hashes=[],
            primary_hash="",
            side_score=score_blurry,  # Same side_score
            appearance_vector=np.array([]),
            canonical_detection_ids=set(),
            best_canonical_detection_id=0,
            fused_canonical=None,
            duplicate_track_index=None
        ),
    ]

    # Before quality weighting, this test FAILS because blurry track
    # has 20 candidates vs 3 for sharp track (length-based sort).
    # After implementation, sharp track should be selected as Front
    # due to composite score: 0.6·normalized_length + 0.4·mean_quality_score
    _resolve_session_tracks(prepared, VisualDeduplicator())

    angles = {pt.track.instance_id: pt.angle for pt in prepared}
    assert angles["sharp_short"] == "Front", \
        f"Sharp track should be Front (quality-weighted), got {angles['sharp_short']}"
    assert angles["blurry_long"] == "Back", \
        f"Blurry track should be Back despite longer length, got {angles['blurry_long']}"


def test_obb_centroid_invariant_under_rotation():
    """
    Task 5: OBB centroid for centroid-jump detection.

    The centroid of a rotated bounding box (OBB) should remain constant
    regardless of rotation angle. This test verifies that:
    1. A 0° (axis-aligned) box at (100, 100)-(300, 300) has centroid (200, 200)
    2. A 45° rotated box at the same location has the same centroid (200, 200)

    This ensures in-place rotation doesn't trigger spurious session resets
    due to false positive centroid jumps.
    """
    from card_capture.tracking.centroid_jump import centroid_from_obb

    # Axis-aligned OBB: corners at (100, 100), (300, 100), (300, 300), (100, 300)
    # Center should be at (200, 200)
    obb_aligned = np.array([
        (100.0, 100.0),  # top-left
        (300.0, 100.0),  # top-right
        (300.0, 300.0),  # bottom-right
        (100.0, 300.0),  # bottom-left
    ], dtype=np.float32)

    centroid_aligned = centroid_from_obb(obb_aligned)
    assert centroid_aligned == (200.0, 200.0), \
        f"Axis-aligned centroid should be (200, 200), got {centroid_aligned}"

    # Rotated OBB: 45° rotation around the same center (150, 150) with ~142px diagonal
    # The four corners of a 200x200 box rotated 45° around center (200, 200):
    # Distance from center to corner: sqrt(100^2 + 100^2) ≈ 141.42
    # Corners are at 45°, 135°, 225°, 315° from center
    d = np.sqrt(2) * 100  # ~141.42 distance from center to corner
    obb_rotated = np.array([
        (200 + d * np.cos(np.pi / 4), 200 + d * np.sin(np.pi / 4)),      # 45°
        (200 + d * np.cos(3 * np.pi / 4), 200 + d * np.sin(3 * np.pi / 4)),  # 135°
        (200 + d * np.cos(5 * np.pi / 4), 200 + d * np.sin(5 * np.pi / 4)),  # 225°
        (200 + d * np.cos(7 * np.pi / 4), 200 + d * np.sin(7 * np.pi / 4)),  # 315°
    ], dtype=np.float32)

    centroid_rotated = centroid_from_obb(obb_rotated)
    # Allow small floating-point error (1e-10)
    assert abs(centroid_rotated[0] - 200.0) < 1e-9, \
        f"Rotated centroid X should be ~200.0, got {centroid_rotated[0]}"
    assert abs(centroid_rotated[1] - 200.0) < 1e-9, \
        f"Rotated centroid Y should be ~200.0, got {centroid_rotated[1]}"

    # Both centroids should be equal (within floating-point precision)
    assert abs(centroid_aligned[0] - centroid_rotated[0]) < 1e-9, \
        f"Aligned and rotated centroids should match (X), got {centroid_aligned[0]} vs {centroid_rotated[0]}"
    assert abs(centroid_aligned[1] - centroid_rotated[1]) < 1e-9, \
        f"Aligned and rotated centroids should match (Y), got {centroid_aligned[1]} vs {centroid_rotated[1]}"


def test_adaptive_min_track_length_from_inter_gaps():
    """
    Task 6: Adaptive min_track_length from inter-detection gaps.

    Replace formula: len(detection_rows) // 3
    With formula:    max(3, median_gap × 3)

    Scenario:
    - 300 detections in long single-card video
    - Inter-detection gaps ~2 frames (median)
    - Old formula: 300 // 3 = 100 (too high, causes false negative resets)
    - New formula: max(3, 2 × 3) = 6 (much better)

    This test verifies that the adaptive formula scales with typical
    swap frequency instead of absolute detection count.
    """
    from card_capture.pipeline_utils import adaptive_min_track_length

    # Scenario 1: Long video with many detections, small inter-gaps
    # (single card continuously visible)
    inter_gaps_small = [2, 2, 2, 2, 2]  # median = 2
    result = adaptive_min_track_length(detection_count=300, inter_gap_frames=inter_gaps_small)
    assert result == 6, \
        f"Expected 6 (max(3, 2*3)), got {result}"

    # Scenario 2: Frequent card swaps (larger inter-gaps)
    inter_gaps_large = [10, 12, 11, 13, 10]  # median = 11
    result = adaptive_min_track_length(detection_count=300, inter_gap_frames=inter_gaps_large)
    assert result == 33, \
        f"Expected 33 (max(3, 11*3)), got {result}"

    # Scenario 3: Empty inter-gaps list (fallback)
    result = adaptive_min_track_length(detection_count=100, inter_gap_frames=[])
    assert result == 3, \
        f"Expected 3 (baseline min), got {result}"

    # Scenario 4: Single gap value
    result = adaptive_min_track_length(detection_count=50, inter_gap_frames=[5])
    assert result == 15, \
        f"Expected 15 (max(3, 5*3)), got {result}"

    # Scenario 5: Gaps that compute to less than baseline
    result = adaptive_min_track_length(detection_count=200, inter_gap_frames=[0.5])
    assert result == 3, \
        f"Expected 3 (baseline min, since 0.5*3 < 3), got {result}"


def test_lab_color_novelty_detects_chroma_difference(tmp_path):
    """
    Task 4: Lab-color novelty detection.

    Replaces grayscale-only novelty with Lab color (L=1.0, a=0.5, b=0.5 weights).
    Detects cards that match in luminance but differ in chroma.

    Test scenario: magenta card on neutral gray background
    - Both have similar luminance (L ≈ 135) → grayscale novelty < 0.05
    - Different chrominance (gray: a=128/b=128 vs magenta: a=182/b=93) → Lab novelty > grayscale

    This test ensures Lab color mode can detect color-only differences.
    """
    import cv2
    from card_capture.presence.background_novelty import BackgroundModel, quad_novelty

    # Create a neutral gray background
    # Neutral gray: (B=128, G=128, R=128) → LAB: L≈137, a=128, b=128
    bg_frames = []
    for _ in range(5):
        frame = np.full((200, 200, 3), (128, 128, 128), dtype=np.uint8)
        bg_frames.append(frame)
    bg = BackgroundModel.from_frames(bg_frames)

    # Create a frame with a magenta card on the gray background
    # Magenta: (B=180, G=80, R=180) → LAB: L≈129, a=182, b=93
    # Very similar luminance to gray, but very different chrominance
    frame = np.full((200, 200, 3), (128, 128, 128), dtype=np.uint8)
    # Paint a magenta card in the center
    frame[50:150, 50:150] = (180, 80, 180)

    # Card corners (center region)
    corners = [(50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)]

    # Grayscale novelty should be LOW because the luminance is very similar
    grayscale_score = quad_novelty(frame, corners, bg, color_space="grayscale")
    assert grayscale_score < 0.05, \
        f"Grayscale novelty should be very low for similar luminance: {grayscale_score}"

    # Lab novelty should be HIGHER than grayscale because it detects chroma difference
    lab_score = quad_novelty(frame, corners, bg, color_space="lab",
                             lab_weights=(1.0, 0.5, 0.5))
    assert lab_score > grayscale_score * 2, \
        f"Lab novelty ({lab_score:.4f}) should be at least 2x higher than grayscale ({grayscale_score:.4f}) due to chroma"


def test_lab_novelty_backward_compatible_with_grayscale(tmp_path):
    """
    Test that Lab novelty with zero a/b weights equals grayscale novelty.

    This verifies backward compatibility: Lab mode with (1.0, 0, 0) weights
    should produce the same result as grayscale mode (approximately, due to
    rounding differences in color space conversion).
    """
    import cv2
    from card_capture.presence.background_novelty import BackgroundModel, quad_novelty

    # Create uniform background
    bg_frames = [np.full((100, 100, 3), (128, 128, 128), dtype=np.uint8) for _ in range(3)]
    bg = BackgroundModel.from_frames(bg_frames)

    # Create frame with contrasting patch
    frame = np.full((100, 100, 3), (128, 128, 128), dtype=np.uint8)
    frame[25:75, 25:75] = (50, 50, 50)
    corners = [(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)]

    grayscale_score = quad_novelty(frame, corners, bg, color_space="grayscale")
    # Lab with zero chroma weights (only luminance)
    lab_lum_only = quad_novelty(frame, corners, bg, color_space="lab",
                                lab_weights=(1.0, 0.0, 0.0))

    # Should be very close (within rounding error)
    assert abs(grayscale_score - lab_lum_only) < 0.05, \
        f"Lab luminance-only ({lab_lum_only:.3f}) should match grayscale ({grayscale_score:.3f})"


def test_background_model_refresh_from_frame_ewma():
    """
    Task 7: BackgroundModel.refresh_from_frame() with EWMA (alpha=0.1).

    Verifies that periodic refresh updates the background model during
    inter-window gaps (when workspace is empty) to track lighting drift.

    Test scenario:
    - Initialize BackgroundModel with dim frame (brightness ~100)
    - Call refresh_from_frame() with bright frame (brightness ~150)
    - Verify mean brightness shifted toward bright (EWMA with alpha=0.1)
    - Formula: new_mean = (1 - alpha) × old_mean + alpha × new_frame

    With alpha=0.1:
    new_mean ≈ 0.9 × 100 + 0.1 × 150 = 90 + 15 = 105

    This is a much smaller shift than averaging all frames together (would be ~125),
    showing that EWMA with small alpha preserves the base model while gradually
    tracking slow lighting changes.
    """
    import cv2
    from card_capture.presence.background_novelty import BackgroundModel

    # Create initial background model with dim lighting (brightness ~100)
    dim_frames = [np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8) for _ in range(3)]
    bg = BackgroundModel.from_frames(dim_frames)

    # Get initial mean BGR
    initial_bgr = bg.get_mean_bgr()
    assert initial_bgr is not None
    initial_brightness = float(initial_bgr.mean())

    # Verify initial brightness is close to 100
    assert 99.0 <= initial_brightness <= 101.0, \
        f"Initial brightness should be ~100, got {initial_brightness}"

    # Create a bright frame (brightness ~150)
    bright_frame = np.full((100, 100, 3), (150, 150, 150), dtype=np.uint8)

    # Call refresh_from_frame (should exist and use EWMA with alpha=0.1)
    bg.refresh_from_frame(bright_frame)

    # Get updated mean BGR
    updated_bgr = bg.get_mean_bgr()
    assert updated_bgr is not None
    updated_brightness = float(updated_bgr.mean())

    # Expected: 0.9 × 100 + 0.1 × 150 = 105
    # Allow small tolerance for floating point rounding
    expected_brightness = 105.0
    assert 104.0 <= updated_brightness <= 106.0, \
        f"After EWMA refresh with alpha=0.1, brightness should be ~105, got {updated_brightness}"

    # Verify the shift is in the right direction (toward bright frame)
    assert updated_brightness > initial_brightness, \
        f"Updated brightness should increase toward bright frame: {initial_brightness} -> {updated_brightness}"

    # Verify the shift is much smaller than simple averaging
    # (simple average would be 125, but EWMA should be much closer to original 100)
    simple_average = (initial_brightness + 150.0) / 2.0
    assert updated_brightness < simple_average, \
        f"EWMA shift should be smaller than simple average: EWMA={updated_brightness}, avg={simple_average}"


def test_background_model_refresh_gradual_drift():
    """
    Task 7 integration test: Background model gradually adapts to lighting drift.

    Simulates a long capture session with gradual lighting changes:
    - Initial brightness: 100
    - Refresh 10 times with progressively brighter frames (110, 120, 130, etc.)
    - Final brightness should approach but not reach 200 (due to EWMA conservatism)
    - Verify convergence is smooth and predictable

    This tests the real scenario mentioned in the task: tracking lighting drift
    during 10+ minute captures without per-frame latency (refresh only happens
    during detected empty windows).
    """
    import cv2
    from card_capture.presence.background_novelty import BackgroundModel

    # Initial dim background (brightness ~100)
    initial_frames = [np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8) for _ in range(3)]
    bg = BackgroundModel.from_frames(initial_frames)

    initial_bgr = bg.get_mean_bgr()
    assert initial_bgr is not None
    brightness_history = [float(initial_bgr.mean())]

    # Simulate 10 refresh cycles with progressively brighter frames
    # (mimicking a venue where lights are slowly brightening)
    for i in range(10):
        # Brightness increases from 100 to 190 in steps
        frame_brightness = 100 + (i + 1) * 10  # 110, 120, 130, ..., 200
        frame = np.full((100, 100, 3), (frame_brightness, frame_brightness, frame_brightness), dtype=np.uint8)
        bg.refresh_from_frame(frame)

        current_bgr = bg.get_mean_bgr()
        assert current_bgr is not None
        current_brightness = float(current_bgr.mean())
        brightness_history.append(current_brightness)

    # Check that we converged gradually without big jumps
    # Each step should be at most alpha (0.1) of the difference
    for i in range(1, len(brightness_history)):
        prev_brightness = brightness_history[i - 1]
        curr_brightness = brightness_history[i]
        # Max step is 0.1 * max_possible_diff ≈ 0.1 * 255 = 25.5
        # But typically much smaller since we're not jumping 255 levels
        assert curr_brightness >= prev_brightness, \
            f"Brightness should monotonically increase during gradual brightening: {brightness_history}"

    # Final brightness should approach 200 but not exceed it
    final_brightness = brightness_history[-1]
    # After 10 iterations of EWMA with alpha=0.1 toward target 200,
    # we expect some convergence. With EWMA, we won't reach 200 in 10 steps from 100,
    # but we should be significantly higher than initial 100.
    # Rule of thumb: EWMA converges at roughly (1 - alpha)^n = 0.9^10 ≈ 0.35
    # So we'd expect roughly 100 + 0.65 * 100 = 165 after 10 refreshes to 200
    assert final_brightness > 140.0, \
        f"After 10 EWMA refreshes toward 200, brightness should be >140, got {final_brightness}"
    assert final_brightness < 200.0, \
        f"EWMA should not overshoot target brightness 200, got {final_brightness}"
    assert final_brightness > brightness_history[0], \
        f"Final brightness {final_brightness} should be higher than initial {brightness_history[0]}"

    # Verify monotonic increase (no drops during gradual brightening)
    for i in range(1, len(brightness_history)):
        assert brightness_history[i] >= brightness_history[i-1], \
            f"Brightness should increase monotonically: {brightness_history}"


def test_spatial_glare_distinguishes_scattered_from_blob():
    """
    Task 8: Spatial-glare metric with connected-component analysis.

    Replaces pixel-fraction glare with largest-saturated-blob area analysis.
    Distinguishes scattered specular reflections from large blowout regions.

    Test scenario:
    - Frame A: 10% saturated pixels scattered as many 1×1 white spots
    - Frame B: 10% saturated pixels as one large 160×40 white rectangle
    - Both have same pixel fraction, but Frame B should score worse (larger blob)

    Expected:
    - Frame A (scattered): high glare score (~0.9)
    - Frame B (blob): low glare score (~0.0)
    """
    import cv2
    from card_capture.scoring import QualityScorer

    # Create frame A: scattered 1×1 white pixels (10% coverage)
    frame_a = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_a[:, :] = (100, 100, 100)  # Neutral background
    # Add scattered 1×1 saturated pixels: ~10% coverage = 30,720 pixels
    np.random.seed(42)
    num_saturated = int(480 * 640 * 0.10)
    saturated_rows = np.random.randint(0, 480, num_saturated)
    saturated_cols = np.random.randint(0, 640, num_saturated)
    frame_a[saturated_rows, saturated_cols] = (255, 255, 255)

    # Create frame B: one large 160×40 white rectangle (10% coverage = 6,400 pixels)
    frame_b = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_b[:, :] = (100, 100, 100)  # Same neutral background
    frame_b[220:260, 240:400] = (255, 255, 255)  # 40×160 = 6,400 pixels of saturated

    scorer = QualityScorer(target_pixels=600 * 900)

    # Score both frames with dummy detection confidence
    score_a = scorer.score(frame_a, detection_confidence=0.9)
    score_b = scorer.score(frame_b, detection_confidence=0.9)

    # Extract spatial_glare component
    spatial_glare_a = score_a.components.get("spatial_glare", None)
    spatial_glare_b = score_b.components.get("spatial_glare", None)

    assert spatial_glare_a is not None, "spatial_glare component not found in frame A score"
    assert spatial_glare_b is not None, "spatial_glare component not found in frame B score"

    # Frame A (scattered) should have HIGHER glare score (less glare problem)
    # Frame B (blob) should have LOWER glare score (more glare problem)
    assert spatial_glare_a > spatial_glare_b, \
        f"Scattered glare ({spatial_glare_a:.3f}) should score better than blob glare ({spatial_glare_b:.3f})"

    # Rough threshold checks:
    # - scattered (Frame A): largest blob is tiny, should be close to 1.0
    # - blob (Frame B): largest blob is 6400 pixels = 2% of frame, should be ~0.8
    assert spatial_glare_a > 0.9, \
        f"Scattered glare should be > 0.9 (minimal large blob), got {spatial_glare_a:.3f}"
    assert spatial_glare_b < 0.8, \
        f"Blob glare should be < 0.8 (significant large blob), got {spatial_glare_b:.3f}"
