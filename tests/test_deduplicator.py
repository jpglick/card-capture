import numpy as np
import pytest
from card_capture.deduplicator import VisualDeduplicator
from card_capture.models import TrackState


def test_track_state_has_reid_embedding_field():
    """TrackState must accept reid_embedding kwarg."""
    ts = TrackState(instance_id="test-id", reid_embedding=None)
    assert ts.reid_embedding is None

    embed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ts2 = TrackState(instance_id="test-id", reid_embedding=embed)
    assert ts2.reid_embedding is not None


def test_reid_embedding_duplicate_detected():
    """is_reid_duplicate returns True for nearly identical embeddings."""
    dedup = VisualDeduplicator()
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.99, 0.01, 0.0], dtype=np.float32)
    emb2 /= np.linalg.norm(emb2)
    assert dedup.is_reid_duplicate(emb1, emb2, threshold=0.1) is True


def test_reid_embedding_non_duplicate_detected():
    """is_reid_duplicate returns False for very different embeddings."""
    dedup = VisualDeduplicator()
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert dedup.is_reid_duplicate(emb1, emb2, threshold=0.1) is False


def test_existing_phash_api_unchanged():
    """is_duplicate(hash1, hash2) must still work exactly as before."""
    dedup = VisualDeduplicator(threshold=6)
    identical_hash = "a" * 16
    assert dedup.is_duplicate(identical_hash, identical_hash) is True
