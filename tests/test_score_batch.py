import numpy as np
import pytest
import cv2

# Skip if torch not available
torch = pytest.importorskip("torch")

from card_capture.stages.score.scoring import QualityScorer


def test_score_batch_close_to_single_and_orders_sharpness():
    scorer = QualityScorer()
    
    # Build sharp and blurry BGR images
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    sharp_np = (np.random.rand(1050, 750, 3) * 255).astype(np.uint8)
    
    # Blurry image
    blur_np = cv2.GaussianBlur(sharp_np, (15, 15), 0)
    
    # Existing single scoring
    single_sharp = scorer.score(sharp_np, 0.9)
    single_blur = scorer.score(blur_np, 0.9)
    
    # Batched scoring
    # QualityScorer expects (N, H, W, 3) BGR uint8 torch.Tensor
    batch_t = torch.from_numpy(np.stack([sharp_np, blur_np])).to(scorer._device)
    
    batch_scores = scorer.score_batch(batch_t, [0.9, 0.9])
    
    assert len(batch_scores) == 2
    assert batch_scores[0].total > batch_scores[1].total       # sharp ranks higher
    
    # Check closeness for sharpness component
    # Drift is accepted due to GPU numerics (float32 vs float64)
    assert abs(batch_scores[0].components["sharpness"] - single_sharp.components["sharpness"]) < 0.05
    # Check closeness for total score
    # Now that border_purity is implemented on GPU, we expect much tighter alignment (< 0.02)
    assert abs(batch_scores[0].total - single_sharp.total) < 0.02

def test_score_batch_empty():
    scorer = QualityScorer()
    batch_t = torch.empty((0, 1050, 750, 3), dtype=torch.uint8).to(scorer._device)
    batch_scores = scorer.score_batch(batch_t, [])
    assert batch_scores == []

def test_score_batch_novelty_defaults():
    scorer = QualityScorer()
    img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    batch_t = torch.from_numpy(np.stack([img])).to(scorer._device)
    batch_scores = scorer.score_batch(batch_t, [0.9])
    assert len(batch_scores) == 1
    assert batch_scores[0].components["novelty"] == 1.0
