"""warp_canonical_batch_gpu (GPU-tensor input) matches the numpy path bit-for-bit on CPU."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")

from card_capture.gpu_refinement import KorniaNormalizer


def _img_and_corners():
    img = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)  # H,W,3 BGR
    corners = [(10.0, 10.0), (180.0, 12.0), (185.0, 280.0), (8.0, 275.0)]
    return img, corners


def test_gpu_tensor_warp_matches_numpy():
    norm = KorniaNormalizer(width=750, height=1050, device="cpu")
    img, corners = _img_and_corners()

    from_numpy = norm.warp_canonical_batch([(img, corners)], rotate_180=False)
    from_tensor = norm.warp_canonical_batch_gpu(
        [(torch.from_numpy(img), corners)], rotate_180=False
    )

    assert len(from_numpy) == 1 and len(from_tensor) == 1
    assert from_numpy[0].shape == (1050, 750, 3)
    # Identical input data through the same warp core → identical output.
    assert np.array_equal(from_numpy[0], from_tensor[0])


def test_gpu_tensor_warp_empty_returns_empty():
    norm = KorniaNormalizer(width=750, height=1050, device="cpu")
    assert norm.warp_canonical_batch_gpu([]) == []
