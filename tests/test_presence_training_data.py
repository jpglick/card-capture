import numpy as np
import cv2

from card_capture.presence.training_data import sample_negative_patches, mine_positive_crops


def test_sample_negative_patches_returns_correct_shape():
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    patches = sample_negative_patches(frame, count=4, patch_size=224, rng_seed=42)
    assert len(patches) == 4
    for p in patches:
        assert p.shape == (224, 224, 3)
        assert p.dtype == np.uint8


def test_sample_negative_patches_skips_when_frame_too_small():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    patches = sample_negative_patches(frame, count=4, patch_size=224, rng_seed=42)
    assert patches == []


def test_mine_positive_crops_extracts_card_region():
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    # Simulated detection corners in the center
    corners = [(800, 400), (1100, 400), (1100, 700), (800, 700)]
    crops = mine_positive_crops(frame, [corners], pad_ratio=0.0, target_size=224)
    assert len(crops) == 1
    assert crops[0].shape == (224, 224, 3)
