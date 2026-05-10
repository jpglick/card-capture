from __future__ import annotations

import uuid
from typing import List, Optional

import numpy as np

from ..selector import ScoredCandidate, TrackState
from .bytetrack_adapter import _AdaptedDetection, _xyxy_from_corners


def _import_botsort():
    """Deferred import so module is importable without boxmot installed."""
    try:
        from boxmot import BoTSORT
        return BoTSORT
    except ImportError as exc:
        raise ImportError(
            "BoT-SORT backend requires the 'boxmot' package. "
            "Install it with: pip install 'card-capture[pipeline_v21]'"
        ) from exc


class BoTSORTAdapter:
    """BoT-SORT tracker adapter mirroring ByteTrackAdapter interface.

    Exposes pending_splits: list of frame indices where a ReID identity
    shift was detected (track_id changed for same spatial region).
    """

    def __init__(
        self,
        min_track_length: int = 3,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        reid_distance_threshold: float = 0.6,
    ) -> None:
        BoTSORT = _import_botsort()
        self._BoTSORT = BoTSORT
        self._track_activation_threshold = track_activation_threshold
        self._lost_track_buffer = lost_track_buffer
        self._minimum_matching_threshold = minimum_matching_threshold
        self.reid_distance_threshold = reid_distance_threshold
        self._tracker = BoTSORT(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        self.min_track_length = min_track_length
        self._tracks: dict[int, TrackState] = {}
        self._all_finalized: list[TrackState] = []
        self.pending_splits: list[int] = []

    def reset(self) -> None:
        """Reset tracker state (e.g., between sessions)."""
        self._all_finalized.extend(self._tracks.values())
        self._tracks = {}
        self._tracker = self._BoTSORT(
            track_activation_threshold=self._track_activation_threshold,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=self._minimum_matching_threshold,
        )
        self.pending_splits = []

    def finalized_tracks(self) -> List[TrackState]:
        return list(self._all_finalized)

    def process(self, candidates: List[ScoredCandidate]) -> List[_AdaptedDetection]:
        """Process detections from one frame; returns adapted detections with track_id."""
        if not candidates:
            return []

        from supervision import Detections

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

        det = Detections(
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
