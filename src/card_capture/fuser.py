import cv2
import numpy as np
from typing import List, Tuple, Optional

from card_capture.ecc_registration import register_frames_via_ecc

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
    def select_lighting_diverse_indices(self, images: List[np.ndarray], max_frames: int = 4) -> List[int]:
        if not images:
            return []
        if len(images) == 1:
            return [0]

        h, w = images[0].shape[:2]
        mid_x, mid_y = w / 2, h / 2
        centroids = [find_glare_centroid(img) for img in images]
        sharpness = [calculate_sharpness(img) for img in images]
        quadrants = [[] for _ in range(4)]
        for i, centroid in enumerate(centroids):
            if centroid is None:
                continue
            x, y = centroid
            q = (0 if x < mid_x else 1) + (0 if y < mid_y else 2)
            quadrants[q].append(i)

        selected_indices: List[int] = []
        for q_list in quadrants:
            if q_list:
                selected_indices.append(max(q_list, key=lambda i: sharpness[i]))

        if not selected_indices:
            selected_indices = list(range(min(max_frames, len(images))))

        if len(selected_indices) < min(3, len(images)):
            for i in sorted(range(len(images)), key=lambda idx: sharpness[idx], reverse=True):
                if i not in selected_indices:
                    selected_indices.append(i)
                if len(selected_indices) >= min(max_frames, len(images)):
                    break

        return selected_indices[:max_frames]

    def fuse(self, images: List[np.ndarray]) -> np.ndarray:
        if not images: raise ValueError("No images")
        if len(images) == 1: return images[0]
        selected_indices = self.select_lighting_diverse_indices(images, max_frames=4)
        selected_frames = [images[i] for i in selected_indices]

        # Register frames to eliminate jitter before median fusion
        aligned_frames = register_frames_via_ecc(selected_frames, reference_index=0)

        return np.median(np.stack(aligned_frames), axis=0).astype(np.uint8)
