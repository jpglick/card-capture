from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Tuple, Optional

from .models import QualityScore


@dataclass(frozen=True)
class ScoredCandidate:
    detection_id: int
    timestamp_ms: int
    image_path: str
    score: QualityScore
    corners: List[Tuple[float, float]] | None = None
    frame_index: Optional[int] = None


@dataclass
class TrackState:
    instance_id: str
    candidates: List[ScoredCandidate] = field(default_factory=list)
    last_centroid: Optional[Tuple[float, float]] = None
    last_frame_index: Optional[int] = None
    missed_frames: int = 0
    active: bool = True
    angle: str = "Front"


@dataclass
class SpatialCluster:
    """Represents a unique card instance identified by spatial clustering."""
    candidates: List[ScoredCandidate]


def _calculate_centroid(corners: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate the centroid of 4 corner points.
    
    Args:
        corners: List of [x, y] tuples for the 4 corners
    
    Returns:
        Tuple of (centroid_x, centroid_y)
    """
    if not corners or len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")
    
    avg_x = sum(c[0] for c in corners) / 4.0
    avg_y = sum(c[1] for c in corners) / 4.0
    return (avg_x, avg_y)


def _euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


class CandidateSelector:
    def __init__(
        self,
        group_gap_ms: int = 300,
        max_candidates: int = 10,
        spatial_variance_threshold: float = 75.0,
        frames_per_instance: int = 2,
    ):
        """
        Initialize the CandidateSelector with spatial clustering support.
        
        Args:
            group_gap_ms: Temporal gap threshold in milliseconds (reduced from 1000ms in v2.1)
            max_candidates: Legacy parameter, deprecated (no longer hard-caps selections)
            spatial_variance_threshold: Max centroid distance to group as same card instance (pixels)
            frames_per_instance: How many top-quality frames to select per card instance
        """
        self.group_gap_ms = group_gap_ms
        self.max_candidates = max_candidates
        self.spatial_variance_threshold = spatial_variance_threshold
        self.frames_per_instance = frames_per_instance

    def select(self, candidates: Iterable[ScoredCandidate]) -> List[ScoredCandidate]:
        """
        Select best candidates using spatial clustering to separate card instances.
        
        Algorithm:
        1. Sort detections chronologically
        2. Group using hybrid spatial + temporal logic:
           - If spatial distance exceeds threshold OR time gap exceeds group_gap_ms, start new cluster
           - Otherwise, add to current cluster
        3. For each cluster (unique card instance), select top N by confidence
        4. Return all selections (no global cap)
        
        Args:
            candidates: Iterable of ScoredCandidate objects
        
        Returns:
            List of selected ScoredCandidate objects across all clusters
        """
        ordered = sorted(candidates, key=lambda c: c.timestamp_ms)
        
        if not ordered:
            return []
        
        # Build spatial clusters using hybrid logic
        clusters: List[SpatialCluster] = []
        current_cluster: List[ScoredCandidate] = [ordered[0]]
        prev_centroid = None
        
        if ordered[0].corners:
            prev_centroid = _calculate_centroid(ordered[0].corners)
        
        for candidate in ordered[1:]:
            should_split = False
            
            # Check temporal gap
            time_gap = candidate.timestamp_ms - current_cluster[-1].timestamp_ms
            if time_gap > self.group_gap_ms:
                should_split = True
            
            # Check spatial distance (only if both have corners)
            if not should_split and candidate.corners and prev_centroid is not None:
                current_centroid = _calculate_centroid(candidate.corners)
                spatial_dist = _euclidean_distance(prev_centroid, current_centroid)
                if spatial_dist > self.spatial_variance_threshold:
                    should_split = True
                prev_centroid = current_centroid
            elif candidate.corners:
                prev_centroid = _calculate_centroid(candidate.corners)
            
            if should_split:
                # Close current cluster and start new one
                clusters.append(SpatialCluster(candidates=current_cluster))
                current_cluster = [candidate]
                if candidate.corners:
                    prev_centroid = _calculate_centroid(candidate.corners)
            else:
                current_cluster.append(candidate)
        
        # Don't forget the last cluster
        if current_cluster:
            clusters.append(SpatialCluster(candidates=current_cluster))
        
        # Select top frames per cluster
        selections: List[ScoredCandidate] = []
        for cluster in clusters:
            # Sort by confidence (highest first)
            sorted_cluster = sorted(
                cluster.candidates,
                key=lambda c: (c.score.total, -c.timestamp_ms),
                reverse=True,
            )
            # Take top N frames from this cluster
            selections.extend(sorted_cluster[: self.frames_per_instance])
        
        # Sort final selections by confidence and return
        return sorted(
            selections,
            key=lambda c: c.score.total,
            reverse=True,
        )

def _get_polygon_area(corners: List[Tuple[float, float]]) -> float:
    area = 0.0
    n = len(corners)
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0

def _aspect_ratio(corners: List[Tuple[float, float]]) -> float:
    import numpy as np
    pts = np.array(corners, dtype="float32")
    width_top = np.linalg.norm(pts[1] - pts[0])
    width_bottom = np.linalg.norm(pts[2] - pts[3])
    height_right = np.linalg.norm(pts[2] - pts[1])
    height_left = np.linalg.norm(pts[3] - pts[0])
    width = max(1, int(round(max(width_top, width_bottom))))
    height = max(1, int(round(max(height_right, height_left))))
    return width / height if height > 0 else 1.0
