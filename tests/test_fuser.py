import numpy as np
import cv2
import pytest
from card_capture.fuser import find_glare_centroid

def test_find_glare_centroid_simple():
    # Create a 100x100 black image with a 10x10 white square at (20, 30)
    image = np.zeros((100, 100), dtype=np.uint8)
    image[30:40, 20:30] = 255
    
    centroid = find_glare_centroid(image)
    assert centroid is not None
    # Centroid of x: [20, 29] is 24.5, y: [30, 39] is 34.5
    assert abs(centroid[0] - 24.5) < 1.0
    assert abs(centroid[1] - 34.5) < 1.0

def test_find_glare_centroid_no_glare():
    image = np.zeros((100, 100), dtype=np.uint8)
    centroid = find_glare_centroid(image)
    assert centroid is None

from card_capture.fuser import MultiFrameFuser

def test_fusion_erases_simulated_glare():
    # 750x1050 card
    h, w = 1050, 750
    base_image = np.full((h, w, 3), 128, dtype=np.uint8) # Neutral gray
    
    images = []
    # Create 4 images, each with glare in a different quadrant
    # Q1: TL, Q2: TR, Q3: BL, Q4: BR
    glare_pos = [
        (100, 100), # TL
        (600, 100), # TR
        (100, 900), # BL
        (600, 900)  # BR
    ]
    
    for x, y in glare_pos:
        img = base_image.copy()
        cv2.circle(img, (x, y), 50, (255, 255, 255), -1) # Bright glare
        images.append(img)
        
    fuser = MultiFrameFuser()
    fused = fuser.fuse(images)
    
    # In the fused image, those glare spots should be gone or significantly reduced
    # since for each quadrant we pick a frame that doesn't have glare there.
    for x, y in glare_pos:
        # Check a pixel that was at the center of a glare spot
        # It should be back to ~128 (neutral gray)
        assert np.all(np.abs(fused[y, x].astype(int) - 128) < 10)

def test_find_glare_centroid_percentile():
    # Create a 100x100 image with 128 background and a small 255 spot
    image = np.full((100, 100), 128, dtype=np.uint8)
    image[30:35, 20:25] = 255 # 25 pixels at 255
    # Total pixels = 10000. 25 pixels is 0.25%, well within top 5%.
    
    centroid = find_glare_centroid(image)
    assert centroid is not None
    assert abs(centroid[0] - 22.0) < 1.0
    assert abs(centroid[1] - 32.0) < 1.0
