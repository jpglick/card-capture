from __future__ import annotations

import cv2
import numpy as np

from .models import QualityScore

CARD_ASPECT_RATIO: float = 63.5 / 88.9  # ≈ 0.714 (width / height, standard trading card)
ASPECT_TOLERANCE: float = 0.25


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

        h, w = image.shape[:2]
        actual_ratio = (w / h) if h > 0 else 0.0
        aspect_ratio = _clamp(
            1.0 - abs(actual_ratio - CARD_ASPECT_RATIO) / ASPECT_TOLERANCE
        )

        size = _clamp((h * w) / float(self.target_pixels))

        # Grayscale std-dev rewards textured fronts over plain backs
        complexity = _clamp(float(gray.std()) / 80.0)

        confidence = _clamp(float(detection_confidence))

        total = (
            sharpness * 0.30
            + glare * 0.20
            + aspect_ratio * 0.20
            + size * 0.15
            + complexity * 0.10
            + confidence * 0.05
        )
        components = {
            "sharpness": round(sharpness, 6),
            "glare": round(glare, 6),
            "aspect_ratio": round(aspect_ratio, 6),
            "size": round(size, 6),
            "complexity": round(complexity, 6),
            "confidence": round(confidence, 6),
        }
        return QualityScore(total=round(total, 6), components=components)
