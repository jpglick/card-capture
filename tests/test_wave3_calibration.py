"""Tests for Wave 3 embedding-based same-card criterion and threshold calibration."""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path

from card_capture.identity.embedding_distance import (
    compute_embedding_distance,
    embedding_same_card_score,
)


class TestEmbeddingDistance:
    """Test embedding distance computation for same-card detection."""

    def test_embedding_distance_same_physical_card(self):
        """Same card (Front/Back) should have low distance (<0.5)."""
        # Create two embeddings that are similar (high cosine similarity)
        # Simulating front and back of the same physical card
        emb_front = np.array([1.0, 0.2, 0.1, 0.3, 0.4], dtype=np.float32)
        emb_back = np.array([0.98, 0.21, 0.09, 0.31, 0.41], dtype=np.float32)

        # Normalize for cosine distance
        emb_front = emb_front / np.linalg.norm(emb_front)
        emb_back = emb_back / np.linalg.norm(emb_back)

        distance = compute_embedding_distance(emb_front, emb_back)

        # Should be low for same card
        assert distance < 0.5, f"Same card distance {distance} should be < 0.5"
        assert distance >= 0.0, "Distance should be non-negative"
        assert distance <= 2.0, "Cosine distance should be <= 2"

    def test_embedding_distance_different_cards(self):
        """Different cards should have higher distance (>0.3)."""
        # Create two embeddings that are dissimilar (low cosine similarity)
        # Simulating different physical cards
        emb_card1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_card2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Normalize for cosine distance
        emb_card1 = emb_card1 / np.linalg.norm(emb_card1)
        emb_card2 = emb_card2 / np.linalg.norm(emb_card2)

        distance = compute_embedding_distance(emb_card1, emb_card2)

        # Should be high for different cards
        assert distance > 0.3, f"Different cards distance {distance} should be > 0.3"
        assert distance >= 0.0, "Distance should be non-negative"
        assert distance <= 2.0, "Cosine distance should be <= 2"

    def test_embedding_same_card_score_threshold(self):
        """Test the threshold logic for same-card classification."""
        # Embeddings with low distance (same card)
        emb_a = np.array([1.0, 0.2, 0.1], dtype=np.float32)
        emb_b = np.array([0.99, 0.21, 0.09], dtype=np.float32)
        emb_a = emb_a / np.linalg.norm(emb_a)
        emb_b = emb_b / np.linalg.norm(emb_b)

        is_same = embedding_same_card_score(emb_a, emb_b, threshold=0.5)
        assert is_same is True, "Should classify as same card with default threshold"

        # Embeddings with high distance (different cards)
        emb_c = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_d = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        emb_c = emb_c / np.linalg.norm(emb_c)
        emb_d = emb_d / np.linalg.norm(emb_d)

        is_same = embedding_same_card_score(emb_c, emb_d, threshold=0.5)
        assert is_same is False, "Should classify as different cards with default threshold"

    def test_embedding_distance_orthogonal(self):
        """Orthogonal embeddings should have distance ~2.0 (max)."""
        emb_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Normalize
        emb_x = emb_x / np.linalg.norm(emb_x)
        emb_y = emb_y / np.linalg.norm(emb_y)

        distance = compute_embedding_distance(emb_x, emb_y)

        # Orthogonal vectors have cosine similarity = 0, so distance = 1 - 0 = 1.0
        # But scipy.spatial.distance.cosine returns the actual cosine distance
        assert distance > 0.9, f"Orthogonal embeddings distance {distance} should be close to 1.0"

    def test_embedding_distance_identical(self):
        """Identical embeddings should have distance 0.0."""
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        emb = emb / np.linalg.norm(emb)

        distance = compute_embedding_distance(emb, emb)

        assert distance == pytest.approx(0.0, abs=1e-6), "Identical embeddings should have distance 0"

    def test_embedding_distance_zero_vector_handling(self):
        """Test that zero vectors are handled gracefully."""
        emb_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        emb_valid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_valid = emb_valid / np.linalg.norm(emb_valid)

        # Should handle gracefully - could return NaN or some max value
        # The implementation should decide the appropriate behavior
        distance = compute_embedding_distance(emb_zero, emb_valid)

        # Just verify it doesn't crash and returns a number
        assert isinstance(distance, (float, np.floating)), "Should return a float"


class TestCrossVideoDedup:
    """Test cross-video deduplication via embeddings and pHash."""

    @pytest.mark.skip(reason="_is_reid_duplicate retired with monolith")
    def test_cross_video_dedup_via_embeddings(self):
        """Same card across videos detected via embedding distance.

        Scenario: Two videos with the same physical card.
        - pHash match (Hamming ≤ 22) → candidates
        - Embeddings available and similar (distance < 0.5) → confirm same card
        """
        from card_capture.shared.pipeline_utils import _is_reid_duplicate
        from card_capture.deduplicator import VisualDeduplicator

        deduplicator = VisualDeduplicator()

        # Create embeddings for the same physical card (low distance)
        emb_card_video1 = np.array([1.0, 0.2, 0.1, 0.3, 0.4], dtype=np.float32)
        emb_card_video2 = np.array([0.98, 0.21, 0.09, 0.31, 0.41], dtype=np.float32)

        # Normalize
        emb_card_video1 = emb_card_video1 / np.linalg.norm(emb_card_video1)
        emb_card_video2 = emb_card_video2 / np.linalg.norm(emb_card_video2)

        # Create a simple test image for pHash (both will have similar hashes)
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        phash_1 = deduplicator.compute_phash(test_img)
        phash_2 = deduplicator.compute_phash(test_img)

        # Same image → identical pHash → Hamming distance = 0
        ham_dist = deduplicator.hamming_distance(phash_1, phash_2)
        assert ham_dist <= 22, "pHash should pass pre-filter"

        # Call _is_reid_duplicate with embeddings
        is_duplicate = _is_reid_duplicate(
            phash_1,
            phash_2,
            emb_card_video1,
            emb_card_video2,
            deduplicator,
        )

        assert is_duplicate is True, "Same card with low embedding distance should be detected as duplicate"

    @pytest.mark.skip(reason="_is_reid_duplicate retired with monolith")
    def test_cross_video_dedup_avoids_false_positives(self):
        """Different cards not deduped even with pHash collision.

        Scenario: Two videos with different physical cards.
        - pHash collision (could match) → candidates
        - Embeddings available but dissimilar (distance > 0.5) → NOT same card
        """
        from card_capture.shared.pipeline_utils import _is_reid_duplicate
        from card_capture.deduplicator import VisualDeduplicator

        deduplicator = VisualDeduplicator()

        # Create embeddings for different physical cards (high distance)
        emb_card_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_card_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Normalize
        emb_card_a = emb_card_a / np.linalg.norm(emb_card_a)
        emb_card_b = emb_card_b / np.linalg.norm(emb_card_b)

        # Create a simple test image for pHash (same → collision)
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        phash = deduplicator.compute_phash(test_img)

        # Call _is_reid_duplicate with different embeddings
        is_duplicate = _is_reid_duplicate(
            phash,
            phash,
            emb_card_a,
            emb_card_b,
            deduplicator,
        )

        assert is_duplicate is False, "Different cards with high embedding distance should NOT be marked as duplicate"


class TestThresholdGridSweep:
    """Test threshold auto-sweep harness for Pareto-optimal points."""

    def test_grid_sweep_finds_pareto_optimal_points(self):
        """Grid sweep over thresholds returns results with pareto_front.

        Scenario: Sweep over novelty_thresholds and hamming_thresholds,
        compute robustness metrics for each combination, identify Pareto-optimal
        (non-dominated) operating points.
        """
        import sys
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from calibrate_wave3 import grid_sweep_thresholds

        # Define small grid for testing
        novelty_thresholds = [0.05, 0.06, 0.07]
        hamming_thresholds = [18, 19, 20]

        # Create minimal synthetic metric corpus
        # (In real use, this would be regression test data with ground truth)
        metric_corpus = {
            "test_video_1": {
                "expected_cards": 10,
                "detected_cards": 9,
                "phantom_count": 2,
                "pipeline_output_count": 11,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.json"

            # Run grid sweep
            result = grid_sweep_thresholds(
                novelty_thresholds=novelty_thresholds,
                hamming_thresholds=hamming_thresholds,
                metric_corpus=metric_corpus,
                output_file=str(output_file),
            )

            # Verify result structure
            assert isinstance(result, dict), "Result should be a dictionary"
            assert "all_points" in result, "Result should contain 'all_points'"
            assert "pareto_front" in result, "Result should contain 'pareto_front'"

            all_points = result["all_points"]
            pareto_front = result["pareto_front"]

            # Should have 3 * 3 = 9 points
            assert len(all_points) == 9, f"Expected 9 points, got {len(all_points)}"

            # Pareto front should have at least 1 point
            assert len(pareto_front) >= 1, "Pareto front should have at least 1 point"

            # Each point should have threshold values and metrics
            for point in all_points:
                assert "novelty_threshold" in point
                assert "hamming_threshold" in point
                assert "metrics" in point
                assert "f1" in point

            # Each pareto point should also be in all_points
            pareto_ids = {(p["novelty_threshold"], p["hamming_threshold"]) for p in pareto_front}
            all_ids = {(p["novelty_threshold"], p["hamming_threshold"]) for p in all_points}
            assert pareto_ids.issubset(all_ids), "Pareto points should be subset of all points"

            # Output file should be written
            assert output_file.exists(), "Output file should be created"

            # Read and verify JSON structure
            with open(output_file) as f:
                saved = json.load(f)
            assert "all_points" in saved
            assert "pareto_front" in saved

    def test_compute_pareto_front_simple(self):
        """Compute Pareto front identifies non-dominated points.

        Scenario: Given 3 points on objectives [F1, recall], identify which
        are non-dominated (maximize both).
        """
        import sys
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from calibrate_wave3 import compute_pareto_front

        points = [
            {"novelty_threshold": 0.05, "f1": 0.80, "recall": 0.75},  # Dominated
            {"novelty_threshold": 0.06, "f1": 0.85, "recall": 0.80},  # Pareto
            {"novelty_threshold": 0.07, "f1": 0.82, "recall": 0.85},  # Pareto
        ]
        objectives = {"f1": "max", "recall": "max"}

        pareto = compute_pareto_front(points, objectives)

        # Should have 2 non-dominated points (0.06 and 0.07)
        assert len(pareto) == 2, f"Expected 2 Pareto points, got {len(pareto)}"

        # Both should be in result
        result_thresholds = {p["novelty_threshold"] for p in pareto}
        assert result_thresholds == {0.06, 0.07}, f"Expected thresholds 0.06 and 0.07, got {result_thresholds}"

    def test_compute_pareto_front_no_dominance(self):
        """All points are Pareto-optimal if none dominate others."""
        import sys
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from calibrate_wave3 import compute_pareto_front

        points = [
            {"novelty_threshold": 0.05, "f1": 0.90, "recall": 0.70},  # High F1, low recall
            {"novelty_threshold": 0.06, "f1": 0.75, "recall": 0.90},  # Low F1, high recall
            {"novelty_threshold": 0.07, "f1": 0.80, "recall": 0.80},  # Balanced
        ]
        objectives = {"f1": "max", "recall": "max"}

        pareto = compute_pareto_front(points, objectives)

        # All 3 should be Pareto-optimal (no point dominates all others)
        assert len(pareto) == 3, f"Expected 3 Pareto points, got {len(pareto)}"


class TestHardCaseCapture:
    """Test hard-case capture for active learning."""

    def test_hard_case_multiple_fronts(self):
        """Detect >2 Fronts as hard case."""
        from card_capture.analysis.hard_case_capture import is_hard_case

        # Session with 3 Fronts (multiple_fronts)
        session = {
            "video_id": 123,
            "session_id": "s1",
            "front_tracks": [
                {"track_id": "t1", "angle": "Front"},
                {"track_id": "t2", "angle": "Front"},
                {"track_id": "t3", "angle": "Front"},
            ],
            "hamming_values": [],
            "confidence_scores": [0.9],
            "border_purity_scores": [0.8],
        }

        reason = is_hard_case(session)
        assert reason is not None, "Should detect multiple fronts as hard case"
        assert "multiple_fronts" in reason, f"Reason should mention multiple fronts, got {reason}"

    def test_hard_case_borderline_hamming(self):
        """Detect borderline Hamming distance (within 4 of 22) as hard case."""
        from card_capture.analysis.hard_case_capture import is_hard_case

        # Session with borderline Hamming (within range [18, 26])
        session = {
            "video_id": 123,
            "session_id": "s1",
            "front_tracks": [
                {"track_id": "t1", "angle": "Front"},
                {"track_id": "t2", "angle": "Back"},
            ],
            "hamming_values": [20],  # Within [18, 26]
            "confidence_scores": [0.9],
            "border_purity_scores": [0.8],
        }

        reason = is_hard_case(session)
        assert reason is not None, "Should detect borderline Hamming as hard case"
        assert "borderline_hamming" in reason, f"Reason should mention borderline hamming, got {reason}"


class TestAdaptiveThresholds:
    """Test per-video adaptive threshold computation."""

    def test_adaptive_novelty_threshold_from_distribution(self):
        """Novelty threshold from p50 of in-video candidate distribution, clipped ±20%."""
        from card_capture.calibration.per_video_adaptive import AdaptiveThresholdComputer

        computer = AdaptiveThresholdComputer()
        global_threshold = 0.08

        # Simulate novelty scores from a video: p50 = 0.105 (need 10+ samples)
        novelty_scores = [0.05, 0.07, 0.09, 0.10, 0.105, 0.11, 0.12, 0.15, 0.20, 0.25]

        adaptive_threshold = computer.compute_novelty_threshold(novelty_scores, global_threshold)

        # Expected: p50 = 0.105, clipped to [0.08 * 0.8, 0.08 * 1.2] = [0.064, 0.096]
        # p50 (0.105) is outside [0.064, 0.096], so should clip to 0.096
        expected_min = global_threshold * 0.8  # 0.064
        expected_max = global_threshold * 1.2  # 0.096

        assert adaptive_threshold >= expected_min, f"Threshold {adaptive_threshold} below min {expected_min}"
        assert adaptive_threshold <= expected_max, f"Threshold {adaptive_threshold} above max {expected_max}"

        # Verify it's the p50, clipped
        p50 = np.median(novelty_scores)  # 0.105
        assert adaptive_threshold == pytest.approx(np.clip(p50, expected_min, expected_max), abs=1e-6)

    def test_adaptive_hamming_threshold_from_intra_track_distances(self):
        """Hamming threshold from p75 of intra-track pHash distances, clipped ±10%."""
        from card_capture.calibration.per_video_adaptive import AdaptiveThresholdComputer

        computer = AdaptiveThresholdComputer()
        global_threshold = 22  # _SAME_CARD_HAMMING_MAX

        # Simulate intra-track Hamming distances: p75 = 8.25
        distances = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]

        adaptive_threshold = computer.compute_hamming_threshold(distances, global_threshold)

        # Expected: p75 * 1.2 = 8.25 * 1.2 ≈ 9.9, clipped to [22 * 0.9, 22 * 1.1] = [19.8, 24.2]
        # So should be 9.9 (within clip range)
        expected_min = global_threshold * 0.9  # 19.8
        expected_max = global_threshold * 1.1  # 24.2

        assert adaptive_threshold >= expected_min, f"Threshold {adaptive_threshold} below min {expected_min}"
        assert adaptive_threshold <= expected_max, f"Threshold {adaptive_threshold} above max {expected_max}"

        # Verify it's p75 * 1.2, clipped
        p75 = np.percentile(distances, 75)
        expected_value = p75 * 1.2
        assert adaptive_threshold == pytest.approx(np.clip(expected_value, expected_min, expected_max), abs=1e-6)

    def test_adaptive_novelty_threshold_fallback_to_global(self):
        """Fallback to global threshold when fewer than 10 samples."""
        from card_capture.calibration.per_video_adaptive import AdaptiveThresholdComputer

        computer = AdaptiveThresholdComputer()
        global_threshold = 0.08

        # Fewer than 10 samples
        novelty_scores = [0.05, 0.07, 0.09]

        adaptive_threshold = computer.compute_novelty_threshold(novelty_scores, global_threshold)

        assert adaptive_threshold == global_threshold, "Should fallback to global for <10 samples"

    def test_adaptive_hamming_threshold_fallback_to_global(self):
        """Fallback to global threshold when fewer than 10 samples."""
        from card_capture.calibration.per_video_adaptive import AdaptiveThresholdComputer

        computer = AdaptiveThresholdComputer()
        global_threshold = 22

        # Fewer than 10 samples
        distances = [2.0, 4.0, 6.0]

        adaptive_threshold = computer.compute_hamming_threshold(distances, global_threshold)

        assert adaptive_threshold == global_threshold, "Should fallback to global for <10 samples"

    @pytest.mark.skip(reason="PipelineContext / _filter_candidates_by_novelty retired with monolith")
    def test_novelty_collection_not_double_counted(self):
        """Novelty scores collected once at candidate-level, not twice via track-level.

        Scenario: Process candidates through _filter_candidates_by_novelty (collects scores)
        and later through _prune_empty_workspace_tracks (reads threshold but does NOT collect).
        Verify len(context.observed_novelty_scores) == N (not 2N).
        """
        from card_capture.shared.pipeline_utils import (
            PipelineContext,
            _filter_candidates_by_novelty,
            _prune_empty_workspace_tracks,
            _PreparedTrack,
        )
        from card_capture.stages.novelty.background_novelty import BackgroundModel
        from card_capture.core.models import ScoredCandidate
        from card_capture.core.models import QualityScore
        import cv2

        # Create a simple background model from a constant frame
        bg_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = BackgroundModel(gray=bg_gray, bgr=bg_frame.astype(np.float32))

        # Create N candidate images with distinct novelties
        N = 12
        candidates = []
        temp_images = []

        for i in range(N):
            # Create distinct test images (higher variance = higher novelty)
            img_array = np.ones((100, 100, 3), dtype=np.uint8) * (80 + i * 5)
            # Add some pattern to increase novelty
            img_array[:, :50] = img_array[:, :50] + 20
            temp_path = f"/tmp/test_candidate_{i}.jpg"
            cv2.imwrite(temp_path, img_array)
            temp_images.append(temp_path)

            # Create candidate with corners
            corners = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
            candidate = ScoredCandidate(
                detection_id=i,
                timestamp_ms=i * 100,
                image_path=temp_path,
                score=QualityScore(total=0.9, components={}),
                corners=corners,
                frame_index=i,
            )
            candidates.append(candidate)

        # Initialize context
        context = PipelineContext()

        # Step 1: Filter candidates (should collect N novelty scores)
        filtered = _filter_candidates_by_novelty(
            candidates, bg, threshold=0.05, context=context
        )

        count_after_filter = len(context.observed_novelty_scores)
        assert count_after_filter == N, (
            f"After filter, expected {N} novelty scores, got {count_after_filter}"
        )

        # Step 2: Create prepared tracks from filtered candidates
        # Simulate track creation (in real pipeline, tracks come from tracking stage)
        prepared_tracks = []
        for candidate in filtered:
            # Create a mock track with one candidate
            from unittest.mock import Mock
            mock_track = Mock()
            mock_track.candidates = [candidate]
            mock_track.instance_id = f"track_{candidate.detection_id}"

            prep = _PreparedTrack(
                track=mock_track,
                session_id=0,
                first_frame_index=candidate.frame_index,
                angle="Front",
                frame_entries=[],
                canonical_entries=[],
                candidate_hashes=[],
                primary_hash="test_hash",
                side_score=0.5,
                appearance_vector=np.zeros(272),
                canonical_detection_ids=set(),
                best_canonical_detection_id=candidate.detection_id,
                fused_canonical=np.zeros((100, 100, 3), dtype=np.uint8),
                embedding=None,
            )
            prepared_tracks.append(prep)

        # Step 3: Prune tracks via track-level gate (should NOT collect again)
        pruned = _prune_empty_workspace_tracks(
            prepared_tracks, bg, threshold=0.05, context=context
        )

        count_after_prune = len(context.observed_novelty_scores)

        # Verify: should still be N (not 2N)
        assert count_after_prune == count_after_filter, (
            f"After prune, expected {count_after_filter} novelty scores "
            f"(not doubled to {count_after_filter * 2}), got {count_after_prune}"
        )

        # Cleanup
        for path in temp_images:
            import os
            try:
                os.remove(path)
            except Exception:
                pass
