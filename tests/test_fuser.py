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
