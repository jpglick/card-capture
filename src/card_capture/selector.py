from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Optional

from .models import QualityScore


@dataclass(frozen=True)
class ScoredCandidate:
    detection_id: int
    timestamp_ms: int
    image_path: str
    score: QualityScore
    corners: List[Tuple[float, float]] | None = None


@dataclass
class TrackState:
    instance_id: str
    candidates: List[ScoredCandidate] = field(default_factory=list)
    last_centroid: Optional[Tuple[float, float]] = None
    last_frame_index: Optional[int] = None
    active: bool = True
    angle: str = "Front"


class HysteresisTracker:
    def __init__(
        self,
        t_high: float = 0.55,
        t_low: float = 0.20,
        max_dist: float = 75.0,
        min_track_length: int = 3,
        max_gap_frames: int = 10,
    ):
        self.t_high = t_high
        self.t_low = t_low
        self.max_dist = max_dist
        self.min_track_length = min_track_length
        self.max_gap_frames = max_gap_frames
        self.active_tracks: List[TrackState] = []

    def process(self, candidate: ScoredCandidate):
        if not candidate.corners:
            return

        import statistics
        centroid = _calculate_centroid(candidate.corners)
        best_track = None
        min_dist = float('inf')

        for track in self.active_tracks:
            if not track.active or track.last_centroid is None:
                continue

            if (
                candidate.frame_index is not None
                and track.last_frame_index is not None
                and candidate.frame_index - track.last_frame_index > self.max_gap_frames
            ):
                continue

            dist = _euclidean_distance(centroid, track.last_centroid)
            if dist < self.max_dist and dist < min_dist:
                min_dist = dist
                best_track = track

        if best_track and candidate.score.total > self.t_low:
            gap_frames = 1
            if candidate.frame_index is not None and best_track.last_frame_index is not None:
                gap_frames = max(1, candidate.frame_index - best_track.last_frame_index)

            gap_flip = 1 < gap_frames <= self.max_gap_frames
            if gap_flip or self.detect_flip(best_track, candidate):
                best_track.active = False
                next_angle = "Back" if best_track.angle == "Front" else "Front"
                self.active_tracks.append(
                    TrackState(
                        instance_id=str(uuid.uuid4()),
                        candidates=[candidate],
                        last_centroid=centroid,
                        last_frame_index=candidate.frame_index,
                        angle=next_angle,
                    )
                )
            else:
                best_track.candidates.append(candidate)
                best_track.last_centroid = centroid
                best_track.last_frame_index = candidate.frame_index
        elif candidate.score.total > self.t_high:
            new_track = TrackState(
                instance_id=str(uuid.uuid4()),
                candidates=[candidate],
                last_centroid=centroid,
                last_frame_index=candidate.frame_index,
                active=True
            )
            self.active_tracks.append(new_track)

    def detect_flip(self, track: TrackState, candidate: ScoredCandidate) -> bool:
        if not candidate.corners or not track.candidates:
            return False

        import statistics
        areas = [_get_polygon_area(c.corners) for c in track.candidates if c.corners]
        if not areas:
            return False

        median_area = statistics.median(areas)
        current_area = _get_polygon_area(candidate.corners)
        current_ratio = _aspect_ratio(candidate.corners)

        recent = track.candidates[-10:]
        collapsed = any(
            c.corners and _get_polygon_area(c.corners) <= 0.2 * median_area for c in recent
        )
        median_ratio = statistics.median([_aspect_ratio(c.corners) for c in recent if c.corners])
        ratio_changed = abs(current_ratio - median_ratio) > 0.08
        expanded = current_area >= 0.8 * median_area
        return collapsed and expanded and ratio_changed

    def finalize(self) -> List[TrackState]:
        return [t for t in self.active_tracks if len(t.candidates) >= self.min_track_length]


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
