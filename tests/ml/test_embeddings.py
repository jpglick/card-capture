"""Tests for the reusable embedding helper.

Closes V4_CONCERNS §1.6.
"""
from __future__ import annotations

import numpy as np
import pytest

from card_capture.ml.embeddings import compute_reid_embedding


def test_compute_reid_embedding_returns_float32_vector(monkeypatch):
    import torch
    from unittest.mock import MagicMock
    
    # Mock the DinoEmbedder class used by the helper
    mock_embedder_inst = MagicMock()
    # Mock embed_array to return a deterministic tensor
    def mock_embed(img):
        # vits14 returns 384-dim
        return torch.zeros((1, 384), dtype=torch.float32)
    mock_embedder_inst.embed_array.side_effect = mock_embed
    mock_embedder_inst.variant = "vits14"
    
    monkeypatch.setattr("card_capture.ml.embeddings.DinoEmbedder", lambda **kwargs: mock_embedder_inst)
    monkeypatch.setattr("card_capture.ml.embeddings._SHARED_EMBEDDER", None)

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    vec = compute_reid_embedding(img)
    
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.ndim == 1
    assert vec.size == 384


def test_compute_reid_embedding_is_deterministic(monkeypatch):
    import torch
    from unittest.mock import MagicMock
    
    mock_embedder_inst = MagicMock()
    mock_embedder_inst.embed_array.return_value = torch.ones((1, 384), dtype=torch.float32)
    mock_embedder_inst.variant = "vits14"
    
    monkeypatch.setattr("card_capture.ml.embeddings.DinoEmbedder", lambda **kwargs: mock_embedder_inst)
    monkeypatch.setattr("card_capture.ml.embeddings._SHARED_EMBEDDER", None)

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    vec1 = compute_reid_embedding(img)
    vec2 = compute_reid_embedding(img)
    np.testing.assert_allclose(vec1, vec2, rtol=1e-5)
