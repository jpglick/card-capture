from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .models import CropResult, Point, Polygon


def order_points_clockwise(points: Sequence[Point]) -> Polygon:
    if len(points) != 4:
        raise ValueError("expected exactly four points")
    pts = np.array(points, dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]

    return (
        (float(top_left[0]), float(top_left[1])),
        (float(top_right[0]), float(top_right[1])),
        (float(bottom_right[0]), float(bottom_right[1])),
        (float(bottom_left[0]), float(bottom_left[1])),
    )


class CardCropper:
    def crop(self, image: np.ndarray, polygon: Sequence[Point]) -> CropResult:
        ordered = order_points_clockwise(polygon)
        pts = np.array(ordered, dtype="float32")
        width_top = np.linalg.norm(pts[1] - pts[0])
        width_bottom = np.linalg.norm(pts[2] - pts[3])
        height_right = np.linalg.norm(pts[2] - pts[1])
        height_left = np.linalg.norm(pts[3] - pts[0])
        width = max(1, int(round(max(width_top, width_bottom))))
        height = max(1, int(round(max(height_right, height_left))))

        destination = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(pts, destination)
        crop = cv2.warpPerspective(image, matrix, (width, height))
        return CropResult(image=crop, width=width, height=height, polygon=ordered)


class PrecisionNormalizer:
    def __init__(self, width: int = 750, height: int = 1050, safety_margin: float = 0.015):
        self.width = width
        self.height = height
        self.safety_margin = safety_margin

    def normalize(self, image: np.ndarray, corners: Sequence[Point]) -> np.ndarray:
        # 1. Order corners clockwise
        ordered = order_points_clockwise(corners)
        pts = np.array(ordered, dtype="float32")

        # 2. Compute perspective transform to self.width x self.height
        destination = np.array(
            [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(pts, destination)

        # 3. Warp using cv2.warpPerspective with flags=cv2.INTER_LANCZOS4
        warped = cv2.warpPerspective(
            image, matrix, (self.width, self.height), flags=cv2.INTER_LANCZOS4
        )

        # 4. Apply safety inner crop (remove background bleed):
        crop_w = int(self.width * self.safety_margin)
        crop_h = int(self.height * self.safety_margin)
        
        # Ensure we don't crop everything if margin is too large (though 0.015 is small)
        if crop_h > 0 and crop_w > 0:
            cropped = warped[crop_h:-crop_h, crop_w:-crop_w]
        else:
            cropped = warped

        # 5. Resize back to self.width x self.height using cv2.INTER_LANCZOS4
        normalized = cv2.resize(
            cropped, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4
        )
        
        return normalized
