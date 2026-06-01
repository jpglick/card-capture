from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from card_capture.core.models import CropResult, Point, Polygon


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


def _orient_for_target_canvas(ordered: Polygon, target_width: int, target_height: int) -> Polygon:
    pts = np.array(ordered, dtype="float32")
    target_ratio = float(target_width) / float(target_height) if target_height > 0 else 1.0
    top_edge = float(np.linalg.norm(pts[1] - pts[0]))
    right_edge = float(np.linalg.norm(pts[2] - pts[1]))

    # Enforce portrait mapping for portrait targets:
    # if top edge is longer than right edge, polygon is sideways.
    if target_ratio < 1.0 and top_edge > right_edge:
        pts = np.roll(pts, shift=1, axis=0)

    return (
        (float(pts[0][0]), float(pts[0][1])),
        (float(pts[1][0]), float(pts[1][1])),
        (float(pts[2][0]), float(pts[2][1])),
        (float(pts[3][0]), float(pts[3][1])),
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

        if width > height:
            pts = np.roll(pts, shift=1, axis=0)
            width, height = height, width

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

    def normalize(self, image: np.ndarray, corners: Sequence[Point], rotate_180: bool = True) -> np.ndarray:
        # 1. Try vImage (macOS native)
        from .ml.inference.vimage_warp import vimage_warp_perspective
        vimg = vimage_warp_perspective(image, corners, (self.width, self.height))
        if vimg is not None:
            if rotate_180:
                vimg = cv2.rotate(vimg, cv2.ROTATE_180)
            return vimg

        # 2. Standard OpenCV path
        ordered = order_points_clockwise(corners)
        oriented = _orient_for_target_canvas(ordered, self.width, self.height)
        pts = np.array(oriented, dtype="float32")
        destination = np.array(
            [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(pts, destination)
        warped = cv2.warpPerspective(image, matrix, (self.width, self.height), flags=cv2.INTER_LANCZOS4)
        
        if rotate_180:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        
        crop_w = int(self.width * self.safety_margin)
        crop_h = int(self.height * self.safety_margin)
        cropped = warped[crop_h:self.height-crop_h, crop_w:self.width-crop_w]
        
        return cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
