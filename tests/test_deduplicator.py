import pytest
import numpy as np
import cv2
from card_capture.deduplicator import VisualDeduplicator

@pytest.fixture
def deduplicator():
    return VisualDeduplicator(threshold=4)

@pytest.fixture
def sample_image():
    # Create a random image with some structure (e.g., a circle)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (150, 150), (128, 128, 128), 2)
    return img

def test_identical_images(deduplicator, sample_image):
    hash1 = deduplicator.compute_phash(sample_image)
    hash2 = deduplicator.compute_phash(sample_image.copy())
    
    assert hash1 == hash2
    assert deduplicator.hamming_distance(hash1, hash2) == 0
    assert deduplicator.is_duplicate(hash1, hash2) is True

def test_noisy_image(deduplicator, sample_image):
    # Set seed for reproducibility
    np.random.seed(42)
    hash1 = deduplicator.compute_phash(sample_image)
    
    # Add slight gaussian noise properly
    noise = np.random.normal(0, 2, sample_image.shape)
    noisy_image = np.clip(sample_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    hash2 = deduplicator.compute_phash(noisy_image)
    
    distance = deduplicator.hamming_distance(hash1, hash2)
    assert distance < deduplicator.threshold
    assert deduplicator.is_duplicate(hash1, hash2) is True

def test_different_images(deduplicator, sample_image):
    hash1 = deduplicator.compute_phash(sample_image)
    
    # Create a completely different image
    different_image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.putText(different_image, "DIFFERENT", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    hash2 = deduplicator.compute_phash(different_image)
    
    distance = deduplicator.hamming_distance(hash1, hash2)
    # Different images should have a high Hamming distance
    assert distance >= deduplicator.threshold
    assert deduplicator.is_duplicate(hash1, hash2) is False

def test_border_variation_insensitivity(deduplicator, sample_image):
    # pHash focuses on inner 60%
    hash1 = deduplicator.compute_phash(sample_image)
    
    # Modify the border (outer 20%)
    modified_border = sample_image.copy()
    cv2.rectangle(modified_border, (0, 0), (200, 40), (255, 0, 0), -1) # Top border
    cv2.rectangle(modified_border, (0, 160), (200, 200), (0, 255, 0), -1) # Bottom border
    
    hash2 = deduplicator.compute_phash(modified_border)
    
    # Should be identical because pHash ignores the border
    assert hash1 == hash2
    assert deduplicator.is_duplicate(hash1, hash2) is True

def test_invalid_image(deduplicator):
    with pytest.raises(ValueError):
        deduplicator.compute_phash(None)
    with pytest.raises(ValueError):
        deduplicator.compute_phash(np.array([]))
