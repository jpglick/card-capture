import cv2
import numpy as np
from typing import List, Tuple, Optional

def find_glare_centroid(image: np.ndarray) -> Optional[Tuple[float, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    if moments["m00"] == 0: return None
    return (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))

def calculate_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

class MultiFrameFuser:
    def fuse(self, images: List[np.ndarray]) -> np.ndarray:
        if not images: raise ValueError("No images")
        if len(images) == 1: return images[0]
        h, w = images[0].shape[:2]
        mid_x, mid_y = w / 2, h / 2
        centroids = [find_glare_centroid(img) for img in images]
        sharpness = [calculate_sharpness(img) for img in images]
        quadrants = [[] for _ in range(4)]
        for i, centroid in enumerate(centroids):
            if centroid is None: continue
            x, y = centroid
            q = (0 if x < mid_x else 1) + (0 if y < mid_y else 2)
            quadrants[q].append(i)
        
        selected_indices = set()
        for q_list in quadrants:
            if q_list:
                best_idx = max(q_list, key=lambda i: sharpness[i])
                selected_indices.add(best_idx)
                
        selected_frames = [images[i] for i in selected_indices]
        if len(selected_frames) < 3 and len(images) > len(selected_frames):
            for i in range(len(images)):
                if i not in selected_indices:
                    selected_frames.append(images[i])
                    if len(selected_frames) >= 4: break

        return np.median(np.stack(selected_frames), axis=0).astype(np.uint8)
