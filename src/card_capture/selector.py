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


class HysteresisTracker:
    def __init__(
        self,
        t_high: float = 0.55,
        t_low: float = 0.20,
        max_dist: float = 75.0,
        min_track_length: int = 12,
        max_gap_frames: int = 15,
    ):
        self.t_high = t_high
        self.t_low = t_low
        self.max_dist = max_dist
        self.min_track_length = min_track_length
        self.max_gap_frames = max_gap_frames
        self.active_tracks: List[TrackState] = []
        self.association_events: list[dict[str, Any]] = []

    def process(self, candidate: ScoredCandidate):
        event = {
            "frame_index": candidate.frame_index,
            "timestamp_ms": candidate.timestamp_ms,
            "detection_id": candidate.detection_id,
            "confidence": float(candidate.score.total),
            "t_high": float(self.t_high),
            "t_low": float(self.t_low),
            "max_dist": float(self.max_dist),
            "active_track_count": sum(1 for track in self.active_tracks if track.active),
            "total_track_count": len(self.active_tracks),
            "centroid_x": None,
            "centroid_y": None,
            "nearest_track_id": None,
            "nearest_track_distance": None,
            "assigned_track_id": None,
            "action": "rejected",
            "split_reason": "missing_corners",
            "assigned_track_length": None,
        }
        if not candidate.corners:
            self.association_events.append(event)
            return

        centroid = _calculate_centroid(candidate.corners)
        event["centroid_x"] = float(centroid[0])
        event["centroid_y"] = float(centroid[1])
        best_track = None
        min_dist = float('inf')
        nearest_track = None
        nearest_dist = float('inf')

        for track in self.active_tracks:
            if not track.active or track.last_centroid is None:
                continue

            dist = _euclidean_distance(centroid, track.last_centroid)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_track = track
            if dist < self.max_dist and dist < min_dist:
                min_dist = dist
                best_track = track

        if nearest_track is not None:
            event["nearest_track_id"] = nearest_track.instance_id
            event["nearest_track_distance"] = float(nearest_dist)

        if best_track and candidate.score.total > self.t_low:
            if best_track.missed_frames > 0 and self.detect_flip(best_track, candidate):
                best_track.active = False
                next_angle = "Back" if best_track.angle == "Front" else "Front"
                new_track = TrackState(
                    instance_id=str(uuid.uuid4()),
                    candidates=[candidate],
                    last_centroid=centroid,
                    last_frame_index=candidate.frame_index,
                    missed_frames=0,
                    angle=next_angle,
                )
                self.active_tracks.append(new_track)
                event["assigned_track_id"] = new_track.instance_id
                event["action"] = "new_track"
                event["split_reason"] = "flip_detected"
                event["assigned_track_length"] = len(new_track.candidates)
            else:
                best_track.candidates.append(candidate)
                best_track.last_centroid = centroid
                best_track.last_frame_index = candidate.frame_index
                best_track.missed_frames = 0
                event["assigned_track_id"] = best_track.instance_id
                event["action"] = "assigned_existing"
                event["split_reason"] = "matched_existing_track"
                event["assigned_track_length"] = len(best_track.candidates)
        elif candidate.score.total > self.t_high:
            new_track = TrackState(
                instance_id=str(uuid.uuid4()),
                candidates=[candidate],
                last_centroid=centroid,
                last_frame_index=candidate.frame_index,
                missed_frames=0,
                active=True
            )
            self.active_tracks.append(new_track)
            event["assigned_track_id"] = new_track.instance_id
            event["action"] = "new_track"
            event["split_reason"] = "no_active_track" if nearest_track is None else "spatial_distance"
            event["assigned_track_length"] = len(new_track.candidates)
        else:
            if best_track is not None:
                event["split_reason"] = "below_low_threshold"
            elif nearest_track is not None:
                event["split_reason"] = "spatial_distance_and_below_high_threshold"
            else:
                event["split_reason"] = "below_high_threshold_no_active_track"
        self.association_events.append(event)

    def detect_flip(self, track: TrackState, candidate: ScoredCandidate) -> bool:
        if not candidate.corners or len(track.candidates) < 4:
            return False

        # Look at the last 4 frames for a change in area
        recent = track.candidates[-4:]
        areas = [_get_polygon_area(c.corners) for c in recent if c.corners]
        if not areas:
            return False
            
        # Find the maximum area in this recent window
        max_area = max(areas)
        current_area = _get_polygon_area(candidate.corners)
        
        # Check if current area is significantly different from the max area
        if max_area > 0:
            area_drop = (max_area - current_area) / max_area
            # A 30% drop from the local max indicates a potential flip or occlusion
            # A 50% increase indicates we might have locked onto a larger background object
            if area_drop > 0.30 or area_drop < -0.50:
                print(f"[Stage: Tracking] | Action: Shape Change Detected (Max Area: {max_area:.1f}, New Area: {current_area:.1f}, Drop: {area_drop:.1%})")
                return True
            
        return False

    def finalize(self) -> List[TrackState]:
        # Keep completed tracks even if they were deactivated by a null-state
        # reset or gap expiry. Finalization should filter by evidence length,
        # not by whether the track is still live at the moment the video ends.
        return [t for t in self.active_tracks if len(t.candidates) >= self.min_track_length]

    def reset_active(self) -> None:
        for track in self.active_tracks:
            track.active = False

    def record_reset_event(self, frame_index: int, timestamp_ms: int, reason: str, gap_frames: Optional[int] = None) -> None:
        self.association_events.append(
            {
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "detection_id": None,
                "confidence": None,
                "t_high": float(self.t_high),
                "t_low": float(self.t_low),
                "max_dist": float(self.max_dist),
                "active_track_count": sum(1 for track in self.active_tracks if track.active),
                "total_track_count": len(self.active_tracks),
                "centroid_x": None,
                "centroid_y": None,
                "nearest_track_id": None,
                "nearest_track_distance": None,
                "assigned_track_id": None,
                "action": "reset",
                "split_reason": reason,
                "gap_frames": gap_frames,
                "assigned_track_length": None,
            }
        )

    def tick(self) -> None:
        for track in self.active_tracks:
            if not track.active:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.max_gap_frames:
                track.active = False
                self.association_events.append(
                    {
                        "frame_index": track.last_frame_index,
                        "timestamp_ms": None,
                        "detection_id": None,
                        "confidence": None,
                        "t_high": float(self.t_high),
                        "t_low": float(self.t_low),
                        "max_dist": float(self.max_dist),
                        "active_track_count": sum(1 for active_track in self.active_tracks if active_track.active),
                        "total_track_count": len(self.active_tracks),
                        "centroid_x": None,
                        "centroid_y": None,
                        "nearest_track_id": track.instance_id,
                        "nearest_track_distance": None,
                        "assigned_track_id": None,
                        "action": "deactivate",
                        "split_reason": "max_gap_exceeded",
                        "gap_frames": track.missed_frames,
                        "assigned_track_length": len(track.candidates),
                    }
                )


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
