import cv2
import numpy as np
from typing import List, Tuple, Optional

def find_glare_centroid(image: np.ndarray) -> Optional[Tuple[float, float]]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Threshold to find brightest pixels (approx top 5%)
    # Using a fixed high threshold for glare
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    moments = cv2.moments(thresh)
    if moments["m00"] == 0:
        return None
        
    cX = moments["m10"] / moments["m00"]
    cY = moments["m01"] / moments["m00"]
    return (float(cX), float(cY))
