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
