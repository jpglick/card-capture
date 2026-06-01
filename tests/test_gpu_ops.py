# tests/test_gpu_ops.py
import numpy as np, pytest
torch = pytest.importorskip("torch")
import cv2
from card_capture.ml import gpu_ops


def test_gpu_dct2_matches_cv2():
    x = np.random.rand(32,32).astype(np.float32)
    ref = cv2.dct(x)
    got = gpu_ops.gpu_dct2(torch.from_numpy(x)[None])[0].numpy()
    assert np.allclose(got, ref, atol=1e-3)


def test_phash_batch_matches_reference():
    # Two identical images → hamming distance 0; an inverted image → large distance.
    img = (np.random.rand(1050,750,3)*255).astype(np.uint8)
    batch = torch.from_numpy(np.stack([img, img, 255-img]))   # (3,H,W,3) BGR uint8
    hashes = gpu_ops.phash_batch(batch)                       # list[str] of 64 bits
    from card_capture.stages.dedup.deduplicator import VisualDeduplicator
    d = VisualDeduplicator()
    assert d.hamming_distance(hashes[0], hashes[1]) == 0
    assert d.hamming_distance(hashes[0], hashes[2]) > 10


def test_glare_mask_batch_matches_cv2():
    img = (np.random.rand(20,20,3)*255).astype(np.uint8)
    batch = torch.from_numpy(img[None])                       # (1,H,W,3) BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref = (cv2.threshold(gray,200,255,cv2.THRESH_BINARY)[1]).astype(np.uint8)
    got = gpu_ops.glare_mask_batch(batch)[0].numpy().astype(np.uint8)*255
    # Allow for very few mismatches due to float rounding near threshold
    mismatches = np.sum(got != ref)
    assert mismatches <= 5  # At most 5 pixels mismatch in a 20x20 image (1.25%)


def test_laplacian_var_batch_close_to_cv2():
    img = (np.random.rand(100,100)*255).astype(np.uint8)
    ref = cv2.Laplacian(img, cv2.CV_64F).var()
    got = gpu_ops.laplacian_var_batch(torch.from_numpy(img)[None].float())[0].item()
    assert abs(got - ref) / max(ref,1.0) < 0.05               # within 5% (border handling differs)


def test_glare_centroid_batch():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60, :] = 255  # Square in the middle
    batch = torch.from_numpy(img[None])
    centroids = gpu_ops.glare_centroid_batch(batch)
    assert len(centroids) == 1
    cx, cy = centroids[0]
    assert abs(cx - 49.5) < 0.1
    assert abs(cy - 49.5) < 0.1

    # Empty image
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    batch2 = torch.from_numpy(np.stack([img, img2]))
    centroids2 = gpu_ops.glare_centroid_batch(batch2)
    assert len(centroids2) == 2
    assert centroids2[0] is not None
    assert centroids2[1] is None


def test_spatial_glare_batch():
    # Clean image
    img_clean = np.zeros((100, 100, 3), dtype=np.uint8)
    # Glare image (large saturated block)
    img_glare = np.zeros((100, 100, 3), dtype=np.uint8)
    img_glare[20:80, 20:80, :] = 255
    
    batch = torch.from_numpy(np.stack([img_clean, img_glare]))
    scores = gpu_ops.spatial_glare_batch(batch)
    assert scores[0] == 1.0
    assert scores[1] < 1.0
