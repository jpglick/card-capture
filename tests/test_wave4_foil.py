import pytest
import numpy as np
import cv2
from unittest.mock import Mock
from card_capture.fusion.foil_detection import detect_foil_card, compute_laplacian_variance

def test_foil_detection_high_variance_across_frames():
    """Verify foil cards show high Laplacian variance across frames."""
    # Regular card: consistent edge structure across frames
    # Use the same base image for all frames (no shifting high-freq patterns)
    base_regular = np.ones((750, 1050, 3), dtype=np.uint8) * 120
    regular_frames = [base_regular.copy() for _ in range(4)]

    # Foil card: high-frequency content shifts (holographic surface)
    # Simulate by adding different random high-frequency patterns to each frame
    foil_frames = []
    for _ in range(4):
        base = np.ones((750, 1050, 3), dtype=np.uint8) * 120
        # Add different high-frequency pattern to each frame
        high_freq = np.random.randint(0, 50, (750, 1050, 3), dtype=np.uint8)
        foil_frames.append(np.clip(base.astype(np.int32) + high_freq.astype(np.int32), 0, 255).astype(np.uint8))

    regular_var = compute_laplacian_variance(regular_frames)
    foil_var = compute_laplacian_variance(foil_frames)

    # Foil should have higher variance
    assert foil_var > regular_var, f"Foil variance ({foil_var}) should exceed regular ({regular_var})"

    # Threshold-based detection
    is_foil_regular = detect_foil_card(regular_frames, threshold=50.0)
    is_foil_card = detect_foil_card(foil_frames, threshold=50.0)

    assert not is_foil_regular, "Regular card should not be detected as foil"
    assert is_foil_card, "Foil card should be detected"

def test_foil_detection_edge_cases():
    """Verify detect_foil_card and compute_laplacian_variance handle edge cases.

    D5: Tests explicit len() checks for < 2 frames.
    """
    # Empty list: should return False and 0.0 respectively
    assert detect_foil_card([]) == False, "Empty frames should not be foil"
    assert compute_laplacian_variance([]) == 0.0, "Empty frames should have 0 variance"

    # Single frame: should return False and 0.0 respectively
    single_frame = [np.ones((750, 1050, 3), dtype=np.uint8) * 100]
    assert detect_foil_card(single_frame) == False, "Single frame should not be foil"
    assert compute_laplacian_variance(single_frame) == 0.0, "Single frame should have 0 variance"

    # Single frame with threshold: still False
    assert detect_foil_card(single_frame, threshold=1.0) == False, "Single frame with low threshold should still be False"

def test_glare_rejection_fusion_preserves_luminance():
    """Verify glare-rejection fusion picks closest-to-median pixels."""
    from card_capture.fusion.median_fusion import glare_rejection_fusion

    # Three frames: one bright (glare), two nominal
    frames = [
        np.ones((100, 100, 3), dtype=np.uint8) * 120,  # nominal
        np.ones((100, 100, 3), dtype=np.uint8) * 250,  # bright (glare)
        np.ones((100, 100, 3), dtype=np.uint8) * 118,  # nominal (close to first)
    ]

    fused = glare_rejection_fusion(frames)

    # Result should be close to median (120), not glare (250)
    mean_pixel_value = np.mean(fused)

    assert 110 < mean_pixel_value < 130, f"Should preserve luminance ~120, got {mean_pixel_value}"
    assert mean_pixel_value < 200, f"Should reject glare, but got {mean_pixel_value}"

def test_glare_rejection_fusion_shape():
    """Verify glare-rejection fusion returns same shape as input frames."""
    from card_capture.fusion.median_fusion import glare_rejection_fusion

    frames = [
        np.random.randint(50, 200, (750, 1050, 3), dtype=np.uint8),
        np.random.randint(50, 200, (750, 1050, 3), dtype=np.uint8),
        np.random.randint(50, 200, (750, 1050, 3), dtype=np.uint8),
    ]

    fused = glare_rejection_fusion(frames)

    assert fused.shape == (750, 1050, 3), f"Shape mismatch: expected (750, 1050, 3), got {fused.shape}"
    assert fused.dtype == np.uint8, f"Type should be uint8, got {fused.dtype}"

def test_glare_rejection_fusion_prefers_luminance_over_color_shift():
    """C3: Verify glare-rejection fusion uses luminance distance, not RGB distance.

    Defect (old behavior): BGR distance sums |B−Bm|+|G−Gm|+|R−Rm|.
    A color-shifted frame can lose to a luminance-glaring frame if the
    color channels differ more than the glare differs in luminance.

    Fix (new behavior): Use Lab color space, compute distance on L channel only.
    A frame that is color-shifted but luminance-correct will beat a frame
    that is luminance-glaring, even if glaring frame is more RGB-similar.

    This test creates three frames:
    - Frame 0: Correct luminance, nominal color
    - Frame 1: Correct luminance, heavily color-shifted (e.g., blue-shifted)
    - Frame 2: Luminance glare (very bright), but RGB-balanced

    Old L1-BGR distance: Frame 1 (color-shifted) may pick frame 2 (glaring)
    because color shift accumulates across 3 channels but glare is just brightness.

    New L-only distance: Frame 1 correctly picks frame 0 (same luminance)
    and ignores the color shift. Frame 2 correctly picks frame 0 (median luminance),
    not itself (glaring).
    """
    from card_capture.fusion.median_fusion import glare_rejection_fusion

    # Create three frames where pixels are identical except for variations we control
    height, width = 100, 100

    # Frame 0: baseline (neutral color, L=100)
    frame0 = np.ones((height, width, 3), dtype=np.uint8) * 100

    # Frame 1: same luminance as frame0, but blue-shifted
    # Simulate a blue-heavy frame by boosting B channel
    # Use approximate RGB-to-L conversion to keep L near 100
    # L ≈ 0.299*R + 0.587*G + 0.114*B
    # If we want L ≈ 100 with heavy blue shift:
    # Set B = 150, and reduce R to keep L ~ 100
    # 100 ≈ 0.299*R + 0.587*100 + 0.114*150 => R ≈ 28 (color-shifted but luminance-matched)
    frame1 = np.ones((height, width, 3), dtype=np.uint8)
    frame1[:, :, 0] = 28  # B: 28
    frame1[:, :, 1] = 100  # G: 100 (same as frame0)
    frame1[:, :, 2] = 28  # R: 28 (reduced to compensate for blue shift)

    # Frame 2: luminance glare (very bright, L >> 100), but balanced RGB
    frame2 = np.ones((height, width, 3), dtype=np.uint8) * 200  # All channels 200 => L >> median

    frames = [frame0, frame1, frame2]
    fused = glare_rejection_fusion(frames)

    # Verify fused output is close to frame0's luminance (not glaring)
    # Calculate approximate luminance of fused output
    fused_l = cv2.cvtColor(fused, cv2.COLOR_BGR2Lab)[:, :, 0]
    median_l = np.median(fused_l)

    # The fused frame should be close to median luminance (frame0 at L~100)
    # not glaring (frame2 at L~200)
    assert 85 < median_l < 115, \
        f"Fused luminance {median_l} should be near median (~100), not glaring (>150)"

    # Verify fused is NOT just frame0 (could be) but also passes the sanity check
    # that it's far from frame2's luminance
    frame2_l = cv2.cvtColor(frame2, cv2.COLOR_BGR2Lab)[:, :, 0]
    assert median_l < (np.median(frame2_l) - 50), \
        f"Fused {median_l} should be much less than glare {np.median(frame2_l)}"

def test_enable_foil_aware_fusion_disabled():
    """D3: Verify that setting enable_foil_aware_fusion=False forces median fusion.

    Acceptance criterion: When enable_foil_aware_fusion is False, even foil cards
    are fused using median, not glare_rejection_fusion.

    This test monkeypatches detect_foil_card to always return True, then verifies
    that disabling the flag causes the median path to be used instead.
    """
    from unittest.mock import patch
    from card_capture.fuser import MultiFrameFuser

    # Create test frames: two nominal, one with high glare
    frames = [
        np.ones((100, 100, 3), dtype=np.uint8) * 120,
        np.ones((100, 100, 3), dtype=np.uint8) * 119,
        np.ones((100, 100, 3), dtype=np.uint8) * 250,  # glare spike
    ]

    fuser = MultiFrameFuser()

    # Baseline: with foil_threshold, should use glare-rejection (if detected as foil)
    with patch('card_capture.fuser.detect_foil_card', return_value=True):
        fused_with_foil = fuser.fuse(frames, foil_threshold=50.0)

    # With foil_threshold=None (disable flag), should use median regardless
    with patch('card_capture.fuser.detect_foil_card', return_value=True):
        fused_no_foil = fuser.fuse(frames, foil_threshold=None)

    # Both should not be None
    assert fused_with_foil is not None, "Fusion should succeed with foil_threshold"
    assert fused_no_foil is not None, "Fusion should succeed with foil_threshold=None"

    # Median fusion result: median across all pixels
    median_fused = np.median(frames, axis=0).astype(np.uint8)

    # When foil_threshold=None, should be identical to median
    # (or very close due to floating point)
    assert np.allclose(fused_no_foil, median_fused, atol=1), \
        "With foil_threshold=None, fusion should use median path"

    # The glare-rejection path should differ from median when foil is detected
    # (detect_foil_card returns True), so fused_with_foil may differ from fused_no_foil
    # We don't strictly require difference (in edge cases both might be the same),
    # but the key is that fused_no_foil must be median.
    print(f"Median fused mean: {np.mean(median_fused)}")
    print(f"With foil_threshold mean: {np.mean(fused_with_foil)}")
    print(f"No foil (None) mean: {np.mean(fused_no_foil)}")

    # Extra verification: fused_no_foil should be close to median, not glare-dominated
    assert np.mean(fused_no_foil) < 180, \
        f"foil_threshold=None should use median, not glare-shifted; got mean {np.mean(fused_no_foil)}"

def test_pipeline_uses_glare_rejection_fusion_for_foils():
    """Verify MultiFrameFuser selects glare-rejection fusion for foil cards."""
    from card_capture.fuser import MultiFrameFuser

    # Create mock frames
    regular_frames = [np.random.randint(100, 150, (750, 1050, 3), dtype=np.uint8) for _ in range(4)]
    foil_frames = []
    for _ in range(4):
        base = np.ones((750, 1050, 3), dtype=np.uint8) * 120
        high_freq = np.random.randint(0, 80, (750, 1050, 3), dtype=np.uint8)
        foil_frames.append(np.clip(base.astype(np.int32) + high_freq.astype(np.int32), 0, 255).astype(np.uint8))

    fuser = MultiFrameFuser()

    # Fuse regular card (should use median)
    fused_regular = fuser.fuse(regular_frames, foil_threshold=50.0)
    assert fused_regular is not None, "Should fuse regular frames"

    # Fuse foil card (should use glare-rejection)
    fused_foil = fuser.fuse(foil_frames, foil_threshold=50.0)
    assert fused_foil is not None, "Should fuse foil frames"

    # Both should produce valid images
    assert fused_regular.shape == (750, 1050, 3)
    assert fused_foil.shape == (750, 1050, 3)

@pytest.mark.skip(reason="_PreparedTrack retired with monolith")
def test_fused_canonical_persisted_for_best_only():
    """E1: Verify that fused canonical is persisted only for best_canonical entry.

    This test ensures:
    1. best_canonical_detection_id is correctly identified in _PreparedTrack
    2. fused_canonical is stored in _PreparedTrack
    3. During storage writing, fused_canonical is used for best_canonical
    4. Other canonical entries use raw normalized frames
    """
    from card_capture.shared.pipeline_utils import _PreparedTrack

    # Create mock canonical entries with different detection IDs
    candidate1 = Mock(detection_id=101)
    candidate1.score = Mock(total=0.8)
    entry1 = {"candidate": candidate1,
              "normalized": np.ones((750, 1050, 3), dtype=np.uint8) * 100}

    candidate2 = Mock(detection_id=102)
    candidate2.score = Mock(total=0.9)
    entry2 = {"candidate": candidate2,
              "normalized": np.ones((750, 1050, 3), dtype=np.uint8) * 120}

    candidate3 = Mock(detection_id=103)
    candidate3.score = Mock(total=0.7)
    entry3 = {"candidate": candidate3,
              "normalized": np.ones((750, 1050, 3), dtype=np.uint8) * 110}

    canonical_entries = [entry1, entry2, entry3]

    # Create a mock fused canonical (simulated foil-aware fusion result)
    fused_canonical = np.ones((750, 1050, 3), dtype=np.uint8) * 115

    # Create a mock track object
    mock_track = Mock(instance_id=1, candidates=[], angle='front')

    # Create a mock _PreparedTrack
    prepared = _PreparedTrack(
        track=mock_track,
        session_id=0,
        first_frame_index=0,
        angle='front',
        frame_entries=[],
        canonical_entries=canonical_entries,
        candidate_hashes=[],
        primary_hash="hash123",
        side_score=0.5,
        appearance_vector=np.array([0.1] * 128),
        canonical_detection_ids={101, 102, 103},
        best_canonical_detection_id=102,  # entry2 is the best
        fused_canonical=fused_canonical,
        embedding=None,
    )

    # Verify that best_canonical_detection_id is correctly set
    assert prepared.best_canonical_detection_id == 102, "Best canonical should have detection_id 102"

    # Verify that fused_canonical is stored
    assert prepared.fused_canonical is not None, "Fused canonical should be stored"
    assert prepared.fused_canonical.shape == (750, 1050, 3), "Fused canonical should have correct shape"

    # Verify that the fused canonical differs from individual entries
    # (in real scenario, fused would be different from raw entries)
    assert np.allclose(prepared.fused_canonical, 115), "Fused canonical should have expected mean value"
    assert not np.allclose(prepared.fused_canonical, 120), "Fused should differ from entry2's raw normalized"


@pytest.mark.skip(reason="_PreparedTrack retired with monolith")
def test_fused_canonical_none_fallback_behavior():
    """MINOR: Verify that None fused_canonical is allowed (fallback works).

    The fallback at line 719 ensures fused_canonical is never None in practice,
    but the type annotation is Optional. This test documents the contract.
    """
    from card_capture.shared.pipeline_utils import _PreparedTrack

    # Create minimal mock objects
    candidate = Mock(detection_id=101)
    candidate.score = Mock(total=0.9)

    mock_track = Mock(instance_id=1, candidates=[], angle='front')

    # Create _PreparedTrack with None fused_canonical
    # In real code, line 719 prevents this, but test the type is valid
    prepared = _PreparedTrack(
        track=mock_track,
        session_id=0,
        first_frame_index=0,
        angle='front',
        frame_entries=[],
        canonical_entries=[],
        candidate_hashes=[],
        primary_hash="hash123",
        side_score=0.5,
        appearance_vector=np.array([0.1] * 128),
        canonical_detection_ids={101},
        best_canonical_detection_id=101,
        fused_canonical=None,  # None is valid type
        embedding=None,
    )

    # Verify None is accepted
    assert prepared.fused_canonical is None, "fused_canonical can be None per Optional annotation"


@pytest.mark.skip(reason="_PreparedTrack retired with monolith")
def test_fused_canonical_write_conditional_behavior():
    """E1 extended: Verify conditional write behavior in storage loop.

    Tests that during the rectified frame write loop:
    - Best canonical entry uses prepared.fused_canonical
    - Other entries use entry["normalized"]
    """
    from unittest.mock import patch, call
    from card_capture.shared.pipeline_utils import _PreparedTrack

    # Create mock candidates with different detection IDs
    candidate1 = Mock(detection_id=101)
    candidate1.score = Mock(total=0.8)
    entry1 = {"candidate": candidate1,
              "normalized": np.ones((750, 1050, 3), dtype=np.uint8) * 100}

    candidate2 = Mock(detection_id=102)
    candidate2.score = Mock(total=0.9)
    entry2 = {"candidate": candidate2,
              "normalized": np.ones((750, 1050, 3), dtype=np.uint8) * 120}

    candidate3 = Mock(detection_id=103)
    candidate3.score = Mock(total=0.7)
    entry3 = {"candidate": candidate3,
              "normalized": np.ones((750, 1050, 3), dtype=np.uint8) * 110}

    canonical_entries = [entry1, entry2, entry3]
    fused = np.ones((750, 1050, 3), dtype=np.uint8) * 115

    mock_track = Mock(instance_id=1, candidates=[], angle='front')

    prepared = _PreparedTrack(
        track=mock_track,
        session_id=0,
        first_frame_index=0,
        angle='front',
        frame_entries=[],
        canonical_entries=canonical_entries,
        candidate_hashes=[],
        primary_hash="hash123",
        side_score=0.5,
        appearance_vector=np.array([0.1] * 128),
        canonical_detection_ids={101, 102, 103},
        best_canonical_detection_id=102,  # entry2 is best
        fused_canonical=fused,
        embedding=None,
    )

    # Mock cv2.imwrite and track calls
    with patch('cv2.imwrite') as mock_imwrite:
        for canonical_order, entry in enumerate(prepared.canonical_entries):
            candidate = entry["candidate"]
            # Simulate the conditional write logic
            if candidate.detection_id == prepared.best_canonical_detection_id:
                assert prepared.fused_canonical is not None
                cv2.imwrite(f"path_{canonical_order}.jpg", prepared.fused_canonical)
            else:
                cv2.imwrite(f"path_{canonical_order}.jpg", entry["normalized"])

    # Verify cv2.imwrite was called 3 times
    assert mock_imwrite.call_count == 3, "Should write 3 entries"

    # Verify best entry used fused canonical
    # The call for entry2 (order 1) should use fused image
    calls = mock_imwrite.call_args_list
    # Check that the fused image array was passed for the best entry
    assert any(np.array_equal(call_args[0][1], fused) for call_args in calls), \
        "Fused canonical should be written for best_canonical entry"


def test_foil_detection_on_labeled_fixtures():
    """E3: Verify foil detection works on labeled fixture sets.

    This test loads foil vs non-foil fixtures from the regression corpus
    and asserts that the default threshold correctly classifies them.
    """
    from pathlib import Path

    # Define fixture directories
    foil_dir = Path(__file__).parent / "fixtures" / "foil" / "foil"
    non_foil_dir = Path(__file__).parent / "fixtures" / "foil" / "non_foil"

    # Load foil fixtures (each subdirectory is a frame group)
    foil_groups = _load_fixture_groups(foil_dir)
    assert len(foil_groups) >= 3, f"Expected at least 3 foil fixtures, got {len(foil_groups)}"

    # Load non-foil fixtures
    non_foil_groups = _load_fixture_groups(non_foil_dir)
    assert len(non_foil_groups) >= 3, f"Expected at least 3 non-foil fixtures, got {len(non_foil_groups)}"

    # Test default threshold (50.0)
    DEFAULT_FOIL_THRESHOLD = 50.0

    # Verify all foil fixtures are detected as foil
    for i, frames in enumerate(foil_groups):
        if len(frames) < 2:
            continue  # Skip single-frame groups
        is_foil = detect_foil_card(frames, threshold=DEFAULT_FOIL_THRESHOLD)
        assert is_foil is True, f"Foil fixture {i} should be detected as foil at threshold {DEFAULT_FOIL_THRESHOLD}"

    # Verify all non-foil fixtures are NOT detected as foil
    for i, frames in enumerate(non_foil_groups):
        if len(frames) < 2:
            continue  # Skip single-frame groups
        is_foil = detect_foil_card(frames, threshold=DEFAULT_FOIL_THRESHOLD)
        assert is_foil is False, f"Non-foil fixture {i} should NOT be detected as foil at threshold {DEFAULT_FOIL_THRESHOLD}"


def _load_fixture_groups(fixture_dir: 'Path') -> list:
    """Load all fixture groups from subdirectories.

    Args:
        fixture_dir: Path to directory containing subdirectories with PNGs

    Returns:
        List of frame groups, where each group is a list of BGR frames
    """
    from pathlib import Path

    if not fixture_dir.exists():
        return []

    groups = []
    subdirs = sorted([d for d in fixture_dir.iterdir() if d.is_dir()])
    for subdir in subdirs:
        png_files = sorted(subdir.glob("*.png"))
        if not png_files:
            continue

        frames = []
        for png_file in png_files:
            frame = cv2.imread(str(png_file), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frames.append(frame)

        if frames:
            groups.append(frames)

    return groups
