from __future__ import annotations

import cv2
import numpy as np

from .models import QualityScore


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class QualityScorer:
    def __init__(self, target_pixels: int = 600 * 900):
        self.target_pixels = target_pixels

    def score(self, image: np.ndarray, detection_confidence: float) -> QualityScore:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = _clamp(laplacian_variance / 1000.0)
        overexposed_fraction = float((gray >= 245).mean())
        glare = _clamp(1.0 - overexposed_fraction * 4.0)
        size = _clamp((image.shape[0] * image.shape[1]) / float(self.target_pixels))
        confidence = _clamp(float(detection_confidence))

        total = (
            sharpness * 0.40
            + glare * 0.25
            + size * 0.20
            + confidence * 0.15
        )
        components = {
            "sharpness": round(sharpness, 6),
            "glare": round(glare, 6),
            "size": round(size, 6),
            "confidence": round(confidence, 6),
        }
        return QualityScore(total=round(total, 6), components=components)
