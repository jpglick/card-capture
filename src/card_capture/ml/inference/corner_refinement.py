"""RANSAC-based corner refinement for sub-pixel accurate card extraction.

Fits lines to the edges of the card in the source frame to refine
the coarse YOLO corners.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional
from card_capture.core.models import Point


def refine_corners(
    image: np.ndarray,
    coarse_corners: List[Point],
    roi_width: int = 15,
    ransac_threshold: float = 1.0,
) -> List[Point]:
    """
    Refine corners by fitting lines to the edges near the coarse corners.

    Args:
        image: Source BGR image.
        coarse_corners: List of 4 (x, y) coarse corners from YOLO.
        roi_width: Width of the search region along each edge.
        ransac_threshold: Distance threshold for RANSAC line fitting.

    Returns:
        Refined corners as list of 4 (x, y) tuples.
    """
    from ..cropper import order_points_clockwise
    
    # Ensure clockwise order: TL, TR, BR, BL
    corners = np.array(order_points_clockwise(coarse_corners), dtype=np.float32)
    
    # 1. Fit 4 lines to the edges
    lines = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        
        line = _fit_edge_line(image, p1, p2, roi_width, ransac_threshold)
        if line is None:
            # Fallback if line fitting fails
            return coarse_corners
        lines.append(line)
        
    # 2. Intersect lines to find refined corners
    refined = []
    for i in range(4):
        l1 = lines[(i - 1) % 4] # TL is intersection of BL-TL and TL-TR
        l2 = lines[i]
        
        pt = _intersect_lines(l1, l2)
        if pt is None:
            return coarse_corners
        refined.append(pt)
        
    # 3. Sanity check: don't drift too far (max 20px)
    dist = np.linalg.norm(np.array(refined) - corners, axis=1)
    if np.any(dist > 20.0):
        return coarse_corners
        
    return [(float(p[0]), float(p[1])) for p in refined]


def _fit_edge_line(
    image: np.ndarray, p1: np.ndarray, p2: np.ndarray, roi_width: int, threshold: float
) -> Optional[np.ndarray]:
    """Fit a line to the edge points between p1 and p2."""
    h, w = image.shape[:2]
    
    # Direction and normal vectors
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None
    dir_v = vec / length
    norm_v = np.array([-dir_v[1], dir_v[0]])
    
    # Define a bounding box for the ROI to speed up processing
    roi_pts = np.array([
        p1 + norm_v * roi_width,
        p1 - norm_v * roi_width,
        p2 - norm_v * roi_width,
        p2 + norm_v * roi_width
    ], dtype=np.float32)
    
    rect = cv2.boundingRect(roi_pts)
    rx, ry, rw, rh = rect
    rx, ry = max(0, rx), max(0, ry)
    rw, rh = min(w - rx, rw), min(h - ry, rh)
    
    if rw <= 0 or rh <= 0:
        return None
        
    roi_img = image[ry:ry+rh, rx:rx+rw]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_y, edge_x = np.where(edges > 0)
    if len(edge_x) < 10:
        return None
        
    # Map back to global coordinates
    pts = np.column_stack([edge_x + rx, edge_y + ry]).astype(np.float32)
    
    # Filter points to only those within roi_width of the ideal segment
    vecs = pts - p1
    dists = np.abs(np.dot(vecs, norm_v))
    projs = np.dot(vecs, dir_v)
    
    mask = (dists <= roi_width) & (projs >= -5.0) & (projs <= length + 5.0)
    pts = pts[mask]
    
    if len(pts) < 10:
        return None
        
    # RANSAC line fitting
    # vx, vy, x, y
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    return line


def _intersect_lines(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
    """Compute intersection of two lines (vx, vy, x0, y0)."""
    vx1, vy1, x1, y1 = l1.flatten()
    vx2, vy2, x2, y2 = l2.flatten()
    
    # Solve system:
    # x = x1 + t1*vx1, y = y1 + t1*vy1
    # x = x2 + t2*vx2, y = y2 + t2*vy2
    # => t1*vx1 - t2*vx2 = x2 - x1
    #    t1*vy1 - t2*vy2 = y2 - y1
    
    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    B = np.array([x2 - x1, y2 - y1])
    
    try:
        t = np.linalg.solve(A, B)
        return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1])
    except np.linalg.LinAlgError:
        return None
