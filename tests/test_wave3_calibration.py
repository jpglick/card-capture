"""Tests for Wave 3 embedding-based same-card criterion."""

import numpy as np
import pytest

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

    def test_cross_video_dedup_via_embeddings(self):
        """Same card across videos detected via embedding distance.

        Scenario: Two videos with the same physical card.
        - pHash match (Hamming ≤ 22) → candidates
        - Embeddings available and similar (distance < 0.5) → confirm same card
        """
        from card_capture.pipeline import _is_reid_duplicate
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

    def test_cross_video_dedup_avoids_false_positives(self):
        """Different cards not deduped even with pHash collision.

        Scenario: Two videos with different physical cards.
        - pHash collision (could match) → candidates
        - Embeddings available but dissimilar (distance > 0.5) → NOT same card
        """
        from card_capture.pipeline import _is_reid_duplicate
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
