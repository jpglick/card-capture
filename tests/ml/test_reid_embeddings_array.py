"""Phase 3 — compute_reid_embedding_array equals compute_reid_embedding(path)."""
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.mark.skipif(
    not Path("models").exists(),
    reason="ReID model weights not available locally",
)
def test_compute_reid_embedding_array_matches_path_variant(tmp_path):
    from card_capture.ml.embeddings import (
        compute_reid_embedding, compute_reid_embedding_array,
    )
    img = (np.random.RandomState(1).rand(1050, 750, 3) * 255).astype(np.uint8)
    path = tmp_path / "x.jpg"
    cv2.imwrite(str(path), img)
    re_read = cv2.imread(str(path))

    from_path = compute_reid_embedding(str(path))
    from_array = compute_reid_embedding_array(re_read)
    assert np.allclose(from_path, from_array, atol=1e-5)
