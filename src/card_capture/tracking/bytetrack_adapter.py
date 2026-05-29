from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..models import ScoredCandidate, TrackState, FramePacket


def _xyxy_from_corners(corners) -> np.ndarray:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)


@dataclass
class _AdaptedDetection:
    candidate: ScoredCandidate
    track_id: int
    instance_id: str


class ByteTrackAdapter:
    """Wraps supervision.ByteTrack to consume ScoredCandidate streams.

    The adapter maintains a stable instance_id (UUID string) per ByteTrack track_id
    so downstream pipeline code keeps its existing identifier shape.
    """

    def __init__(
        self,
        min_track_length: int = 3,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ):
        from supervision import ByteTrack, Detections

        self._ByteTrack = ByteTrack
        self._Detections = Detections
        self._track_activation_threshold = track_activation_threshold
        self._lost_track_buffer = lost_track_buffer
        self._minimum_matching_threshold = minimum_matching_threshold
        self._tracker = ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        self.min_track_length = min_track_length
        self._tracks: dict[int, TrackState] = {}  # track_id -> TrackState
        self._all_finalized: list[TrackState] = []

    def reset(self) -> None:
        """Reset tracker state (e.g., between sessions)."""
        self._all_finalized.extend(self._tracks.values())
        self._tracks = {}
        self._tracker = self._ByteTrack(
            track_activation_threshold=self._track_activation_threshold,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=self._minimum_matching_threshold,
        )

    def finalized_tracks(self) -> List[TrackState]:
        return list(self._all_finalized)

    def process(self, candidates: List[ScoredCandidate]) -> List[_AdaptedDetection]:
        """Process detections from one frame; returns adapted detections with track_id."""
        if not candidates:
            return []

        # Build supervision.Detections
        boxes = []
        confidences = []
        valid_candidates: List[ScoredCandidate] = []
        for cand in candidates:
            if not cand.corners:
                continue
            boxes.append(_xyxy_from_corners(cand.corners))
            confidences.append(float(cand.score.total))
            valid_candidates.append(cand)
        if not boxes:
            return []

        det = self._Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
        )
        self._tracker.update_with_detections(det)

        out: List[_AdaptedDetection] = []
        if det.tracker_id is None:
            return out
        for i, track_id in enumerate(det.tracker_id):
            if track_id is None or int(track_id) == -1:
                continue
            tid = int(track_id)
            cand = valid_candidates[i]
            if tid not in self._tracks:
                self._tracks[tid] = TrackState(
                    instance_id=str(uuid.uuid4()),
                    candidates=[],
                    last_centroid=None,
                    last_frame_index=cand.frame_index,
                )
            state = self._tracks[tid]
            state.candidates.append(cand)
            state.last_frame_index = cand.frame_index
            out.append(_AdaptedDetection(candidate=cand, track_id=tid, instance_id=state.instance_id))
        return out

    def finalize(self) -> List[TrackState]:
        """Return all tracks (current + previously reset) above min length."""
        all_tracks = list(self._tracks.values()) + list(self._all_finalized)
        return [t for t in all_tracks if len(t.candidates) >= self.min_track_length]

    def assign(
        self,
        detections: List[dict] | List[ScoredCandidate],
        frames: List[dict] | List[FramePacket],
    ) -> List[TrackState]:
        """Unified entry point for the pipeline 'track' stage.

        Processes all detections frame-by-frame and returns finalized tracks.
        """
        self.reset()
        if not detections:
            return []

        # Group detections by frame_index
        by_frame: dict[int, list[ScoredCandidate]] = {}
        for d in detections:
            if isinstance(d, dict):
                # Convert dict to ScoredCandidate if needed
                from ..models import QualityScore, ScoredCandidate
                cand = ScoredCandidate(
                    detection_id=d.get("detection_id", str(uuid.uuid4())),
                    timestamp_ms=d.get("timestamp_ms", 0),
                    image_path=d.get("image_path", ""),
                    score=QualityScore(total=d.get("novelty_score", 1.0), components={}),
                    corners=d.get("corners", []),
                    frame_index=d.get("frame_index", 0),
                )
            else:
                cand = d
            
            by_frame.setdefault(cand.frame_index, []).append(cand)

        # Process frames in order
        for idx in sorted(by_frame.keys()):
            self.process(by_frame[idx])

        return self.finalize()
