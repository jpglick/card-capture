"""Wave 2 robustness tests: occlusion residual detection, ECC registration, and other improvements."""

import numpy as np
import pytest
import cv2

from card_capture.occlusion_residual import compute_occlusion_residual_score
from card_capture.scoring import QualityScorer
from card_capture.ecc_registration import compute_ecc_warp, register_frames_via_ecc
from card_capture.fuser import MultiFrameFuser


def _clean_card(h=1050, w=750):
    """Synthesize a clean card-like canonical image (grayscale).

    Standard trading card aspect ratio: 750x1050 (w x h).
    White/light border with textured dark interior.
    """
    gray = np.full((h, w), 240, dtype=np.uint8)  # light border
    gray[50:h - 50, 50:w - 50] = np.random.randint(40, 120, (h - 100, w - 100), dtype=np.uint8)
    return gray


def _add_fingertip_blob(image: np.ndarray, y_start=300, y_end=400, x_start=400, x_end=500):
    """Add a dark fingertip-like blob to the interior of an image."""
    result = image.copy()
    result[y_start:y_end, x_start:x_end] = np.random.randint(20, 60, (y_end - y_start, x_end - x_start), dtype=np.uint8)
    return result


def test_occlusion_residual_detects_fingertip():
    """Test that occlusion residual score detects a fingertip blob via median residual."""
    # Create clean canonical frames (prior frames)
    frame1 = _clean_card()
    frame2 = _clean_card()
    frame3 = _clean_card()
    prior_frames = [frame1, frame2, frame3]

    # Create occluded frame (current) with fingertip blob
    occluded = _add_fingertip_blob(frame3.copy(), y_start=300, y_end=400, x_start=400, x_end=500)

    # Compute occlusion score
    score = compute_occlusion_residual_score(occluded, prior_frames)

    # Fingertip blob is localized (< 30% of tiles), so we expect a penalty
    # Score should be <1.0 but not too low (blob is small relative to frame)
    assert score < 1.0, f"Expected occlusion_score < 1.0 for fingertip, got {score}"
    assert score > 0.0, f"Expected occlusion_score > 0.0, got {score}"
    assert score > 0.5, f"Expected reasonable occlusion_score > 0.5 for small blob, got {score}"


def test_occlusion_residual_no_penalty_without_occlusion():
    """Test that a clean frame (without occlusion) gets no penalty."""
    frame1 = _clean_card()
    frame2 = _clean_card()
    frame3 = _clean_card()
    prior_frames = [frame1, frame2, frame3]

    # Current frame is also clean
    current = _clean_card()

    score = compute_occlusion_residual_score(current, prior_frames)

    # Clean frames should score ~1.0 (no occlusion)
    assert score > 0.95, f"Expected clean frame score > 0.95, got {score}"


def test_occlusion_residual_no_prior_frames():
    """Test that missing prior frames returns neutral score (1.0)."""
    current = _clean_card()
    score = compute_occlusion_residual_score(current, [])

    assert score == 1.0, f"Expected neutral score 1.0 with no prior frames, got {score}"


def test_quality_scorer_with_occlusion_component():
    """Test that QualityScorer integrates occlusion score as component 8."""
    scorer = QualityScorer()

    # Create a clean card image (BGR for scoring)
    clean = np.full((1050, 750, 3), 240, dtype=np.uint8)
    clean[50:1000, 50:700] = np.random.randint(40, 120, (950, 650, 3), dtype=np.uint8)

    # Prior frames (grayscale)
    frame1 = _clean_card()
    frame2 = _clean_card()
    frame3 = _clean_card()
    prior_frames = [frame1, frame2, frame3]

    # Score with prior frames
    score_with_prior = scorer.score(clean, detection_confidence=0.8, prior_frames=prior_frames)

    # Check that occlusion component exists
    assert "occlusion" in score_with_prior.components, "occlusion component not found"
    assert score_with_prior.components["occlusion"] > 0.95, f"Expected high occlusion score for clean frame, got {score_with_prior.components['occlusion']}"

    # Score without prior frames (occlusion should default to 1.0)
    score_without_prior = scorer.score(clean, detection_confidence=0.8, prior_frames=None)
    assert "occlusion" in score_without_prior.components
    assert score_without_prior.components["occlusion"] == 1.0, "Expected occlusion=1.0 when no prior frames"


def test_quality_scorer_penalizes_occluded_frame():
    """Test that QualityScorer gives lower total score for occluded frame."""
    scorer = QualityScorer()

    # Create BGR images
    clean_rgb = np.full((1050, 750, 3), 240, dtype=np.uint8)
    clean_rgb[50:1000, 50:700] = np.random.randint(40, 120, (950, 650, 3), dtype=np.uint8)

    occluded_rgb = clean_rgb.copy()
    # Add fingertip-like dark blob to the BGR image
    occluded_rgb[300:400, 400:500] = np.random.randint(20, 60, (100, 100, 3), dtype=np.uint8)

    # Prior frames (grayscale)
    frame1 = _clean_card()
    frame2 = _clean_card()
    frame3 = _clean_card()
    prior_frames = [frame1, frame2, frame3]

    score_clean = scorer.score(clean_rgb, detection_confidence=0.8, prior_frames=prior_frames)
    score_occluded = scorer.score(occluded_rgb, detection_confidence=0.8, prior_frames=prior_frames)

    # Occluded frame should have lower total score
    assert score_clean.total > score_occluded.total, f"Expected clean ({score_clean.total}) > occluded ({score_occluded.total})"
    # Occlusion component should be lower for occluded frame
    assert score_clean.components["occlusion"] > score_occluded.components["occlusion"], (
        f"Expected clean occlusion ({score_clean.components['occlusion']}) "
        f"> occluded ({score_occluded.components['occlusion']})"
    )


def test_per_pixel_bg_tracks_variance():
    """Verify BG model stores per-pixel variance."""
    from card_capture.presence.background_novelty import BackgroundModel

    # Create two frames with different variance patterns
    # Region [0:50, 0:50]: constant at 150 and 145 (small variance)
    # Region [50:100, 50:100]: varies from 128 to 120 (larger variance)
    frame1 = np.ones((100, 100), dtype=np.uint8) * 128
    frame1[0:50, 0:50] = 150

    frame2 = np.ones((100, 100), dtype=np.uint8) * 120
    frame2[0:50, 0:50] = 145

    bg_model = BackgroundModel.from_frames([frame1, frame2])

    assert hasattr(bg_model, 'variance_gray'), "BG model should have variance_gray"
    assert bg_model.variance_gray is not None, "variance_gray should not be None"
    assert bg_model.variance_gray.shape == (100, 100), f"Expected shape (100, 100), got {bg_model.variance_gray.shape}"
    # Region [50:100, 50:100] has larger variance (128 vs 120) than [0:50, 0:50] (150 vs 145)
    assert bg_model.variance_gray[75, 75] > bg_model.variance_gray[25, 25], \
        f"Expected higher variance in region [50:100, 50:100], got {bg_model.variance_gray[75, 75]} vs {bg_model.variance_gray[25, 25]}"


def test_mahalanobis_novelty_with_variance():
    """Verify Mahalanobis novelty accounts for variance."""
    from card_capture.presence.background_novelty import BackgroundModel, quad_novelty

    # Create background: uniform gray at 128
    bg_frame = np.ones((100, 100), dtype=np.uint8) * 128
    bg_model = BackgroundModel.from_frames([bg_frame])

    # Create test frame: uniform gray at 148 (20 level difference)
    card_frame = np.ones((100, 100), dtype=np.uint8) * 148

    # Define a quad covering the entire frame
    corners = [(0, 0), (100, 0), (100, 100), (0, 100)]

    # Compute novelty with variance
    novelty = quad_novelty(card_frame, corners, bg_model, color_space="grayscale", use_variance=True, k=2.0)

    # With uniform background (low variance) and consistent 20-level shift,
    # novelty should be in reasonable range (0.1 < x < 0.9)
    assert 0.05 < novelty < 0.95, \
        f"Expected novelty in (0.05, 0.95) for 20-level shift, got {novelty}"


class TestPerRegionValleyDetection:
    """Tests for per-region valley detection for multi-card swaps."""

    def _create_high_edge_frame(self, h, w):
        """Create a frame with high edge density."""
        frame = np.zeros((h, w), dtype=np.uint8)
        # Add horizontal lines with edges
        for y in range(20, h - 20, 40):
            frame[y:y+10, :] = 255
        # Add vertical lines with edges
        for x in range(20, w - 20, 40):
            frame[:, x:x+10] = 255
        return frame

    def _create_low_edge_frame(self, h, w):
        """Create a frame with low edge density (smooth)."""
        frame = np.full((h, w), 128, dtype=np.uint8)
        return frame

    def test_per_region_valley_detection_multicard_swap(self):
        """Test detecting valleys when card A leaves left, card B enters right."""
        from card_capture.sampler.valley_detection_per_region import per_region_valley_detection

        h, w = 180, 240

        # Create frame sequence:
        # 0-2: Both regions have edges (stable)
        # 3-5: Left region loses edges (valley), right region gains edges (no valley in right)
        # 6-8: Back to normal
        frames = []

        for i in range(9):
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            if i in [0, 1, 2, 6, 7, 8]:
                # Stable state: edges everywhere
                frame[:, :] = self._create_high_edge_frame(h, w)[:, :, np.newaxis]
            else:
                # Swap state (frames 3-5)
                left_third = w // 3
                right_third = 2 * w // 3

                # Left third loses edges (valley)
                frame[:, :left_third] = self._create_low_edge_frame(h, left_third)[:, :, np.newaxis]

                # Middle stays stable
                frame[:, left_third:right_third] = self._create_high_edge_frame(h, right_third - left_third)[:, :, np.newaxis]

                # Right third gains edges
                frame[:, right_third:] = self._create_high_edge_frame(h, w - right_third)[:, :, np.newaxis]

            frames.append(frame)

        # Detect per-region valleys
        valleys = per_region_valley_detection(
            frames,
            grid_rows=3,
            grid_cols=3,
            valley_drop_ratio=0.70,
            min_valley_width_frames=2,
        )

        # Should detect a valley in the left tile region during swap
        assert len(valleys) > 0, "Expected to detect valleys in multi-card swap scenario"

        # Extract unique frame indices
        valley_frames = sorted(set(frame_idx for frame_idx, _, _ in valleys))

        # Should detect valleys in frames 3-5 (the swap frames)
        assert any(f in [3, 4, 5] for f in valley_frames), \
            f"Expected valley frames in [3, 4, 5], got {valley_frames}"

    def test_find_valley_splits_per_region_returns_unique_frames(self):
        """Test that find_valley_splits_per_region returns unique frame indices."""
        from card_capture.sampler.valley_detection_per_region import find_valley_splits_per_region

        h, w = 180, 240

        # Create frames with a clear valley in frame 3
        frames = []
        for i in range(9):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if i in [0, 1, 2, 6, 7, 8]:
                frame[:, :] = self._create_high_edge_frame(h, w)[:, :, np.newaxis]
            else:
                frame[:, :] = self._create_low_edge_frame(h, w)[:, :, np.newaxis]
            frames.append(frame)

        # Get split frames
        splits = find_valley_splits_per_region(
            frames,
            grid_rows=3,
            grid_cols=3,
            valley_drop_ratio=0.70,
            min_valley_width_frames=2,
        )

        # Should have unique frame indices
        assert len(splits) == len(set(splits)), "Expected unique frame indices in splits"

        # Should be sorted
        assert splits == sorted(splits), "Expected splits to be sorted"

    def test_per_region_valley_empty_frames_returns_empty(self):
        """Test that empty frame list returns no valleys."""
        from card_capture.sampler.valley_detection_per_region import per_region_valley_detection

        valleys = per_region_valley_detection(
            [],
            grid_rows=3,
            grid_cols=3,
            valley_drop_ratio=0.70,
            min_valley_width_frames=3,
        )
        assert valleys == [], "Expected empty result for empty frame list"

    def test_per_region_valley_flat_signal_no_valleys(self):
        """Test that a constant high-edge-density signal produces no valleys."""
        from card_capture.sampler.valley_detection_per_region import per_region_valley_detection

        h, w = 180, 240
        frames = [self._create_high_edge_frame(h, w) for _ in range(6)]

        valleys = per_region_valley_detection(
            frames,
            grid_rows=3,
            grid_cols=3,
            valley_drop_ratio=0.70,
            min_valley_width_frames=3,
        )

        # Should have no valleys (constant signal)
        assert len(valleys) == 0, f"Expected no valleys for constant signal, got {valleys}"

    def test_per_region_valley_grayscale_input(self):
        """Test that grayscale frames are handled correctly."""
        from card_capture.sampler.valley_detection_per_region import per_region_valley_detection

        h, w = 180, 240

        # Create grayscale frames (2D arrays)
        frames = []
        for i in range(6):
            if i < 3:
                frame = self._create_high_edge_frame(h, w)
            else:
                frame = self._create_low_edge_frame(h, w)
            frames.append(frame)

        valleys = per_region_valley_detection(
            frames,
            grid_rows=3,
            grid_cols=3,
            valley_drop_ratio=0.70,
            min_valley_width_frames=2,
        )

        # Should handle grayscale input without errors
        assert isinstance(valleys, list), "Expected list result"


def _candidate_with_rotation(detection_id, frame_index, x, y, conf=0.9, w=200, h=300, angle_deg=0):
    """
    Create a ScoredCandidate with rotated corners.

    Args:
        detection_id: unique detection ID
        frame_index: frame index
        x, y: center of card
        conf: confidence score
        w, h: width and height of card
        angle_deg: rotation angle in degrees (0 = axis-aligned)
    """
    # Create axis-aligned corners first
    corners = np.array([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2],
    ], dtype=np.float32)

    # Rotate if needed
    if angle_deg != 0:
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a],
        ], dtype=np.float32)
        corners = corners @ rot_matrix.T

    # Translate to position
    corners[:, 0] += x
    corners[:, 1] += y

    corners_list = [tuple(c) for c in corners]

    return ScoredCandidate(
        detection_id=detection_id,
        timestamp_ms=frame_index * 33,
        image_path="x.jpg",
        score=QualityScore(total=conf, components={"sharpness": conf, "blur": 0.0, "area": 0.5}),
        corners=corners_list,
        frame_index=frame_index,
    )


class TestOrientedIoU:
    """Tests for rotated IoU in multi-card tracking."""

    def test_rotated_iou_same_card_high_iou(self):
        """Two identical cards at same location should have high IoU."""
        card_a = _candidate_with_rotation(0, 0, x=100, y=100, w=200, h=300, angle_deg=0)
        card_b = _candidate_with_rotation(1, 0, x=100, y=100, w=200, h=300, angle_deg=0)

        iou = rotated_iou(card_a.corners, card_b.corners)
        assert iou > 0.95, f"Expected high IoU for identical cards, got {iou}"

    def test_rotated_iou_perpendicular_cards_different_iou(self):
        """Two cards at 90° to each other have ~0.5 IoU."""
        # Card A: axis-aligned at origin
        card_a = _candidate_with_rotation(0, 0, x=100, y=100, w=200, h=300, angle_deg=0)
        # Card B: rotated 90° at same location
        card_b = _candidate_with_rotation(1, 0, x=100, y=100, w=200, h=300, angle_deg=90)

        iou = rotated_iou(card_a.corners, card_b.corners)
        # 90° rotation of same dimensions gives IoU ~0.5
        # (intersection is 200x200, each card is 200x300, union ~80000, intersection 40000)
        assert 0.4 < iou < 0.6, f"Expected ~0.5 IoU for perpendicular cards, got {iou}"

    def test_rotated_iou_45_degree_rotation(self):
        """Two cards with 45° offset should have moderate IoU."""
        card_a = _candidate_with_rotation(0, 0, x=100, y=100, w=200, h=300, angle_deg=0)
        card_b = _candidate_with_rotation(1, 0, x=100, y=100, w=200, h=300, angle_deg=45)

        iou = rotated_iou(card_a.corners, card_b.corners)
        # 45° should be between 0.3 and 0.95
        assert 0.3 < iou < 0.95, f"Expected moderate IoU for 45° offset, got {iou}"

    def test_axis_aligned_iou_from_corners_same_box(self):
        """Two identical axis-aligned boxes should have IoU=1.0."""
        corners_a = [(0, 0), (100, 0), (100, 100), (0, 100)]
        corners_b = [(0, 0), (100, 0), (100, 100), (0, 100)]

        iou = axis_aligned_iou_from_corners(corners_a, corners_b)
        assert abs(iou - 1.0) < 1e-6, f"Expected IoU=1.0 for identical boxes, got {iou}"

    def test_axis_aligned_iou_from_corners_50_percent_overlap(self):
        """Two boxes with 50% overlap should have IoU=0.333..."""
        corners_a = [(0, 0), (100, 0), (100, 100), (0, 100)]
        corners_b = [(50, 0), (150, 0), (150, 100), (50, 100)]

        iou = axis_aligned_iou_from_corners(corners_a, corners_b)
        # Union = 100*100 + 100*100 - 50*100 = 15000
        # Intersection = 50*100 = 5000
        # IoU = 5000/15000 = 0.333...
        assert abs(iou - 1.0/3.0) < 1e-6, f"Expected IoU=0.333, got {iou}"

    def test_axis_aligned_iou_from_corners_no_overlap(self):
        """Two boxes with no overlap should have IoU=0.0."""
        corners_a = [(0, 0), (100, 0), (100, 100), (0, 100)]
        corners_b = [(200, 0), (300, 0), (300, 100), (200, 100)]

        iou = axis_aligned_iou_from_corners(corners_a, corners_b)
        assert iou == 0.0, f"Expected IoU=0.0 for non-overlapping boxes, got {iou}"
