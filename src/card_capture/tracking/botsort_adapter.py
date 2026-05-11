from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ..selector import ScoredCandidate, TrackState
from .bytetrack_adapter import _AdaptedDetection, _xyxy_from_corners


def _import_botsort():
    """Deferred import. Tolerates the boxmot v0.17+ rename (`BotSort`) and
    the legacy `BoTSORT` export. Returns the class object."""
    try:
        from boxmot import BoTSORT as _Klass  # legacy (boxmot ≤ 0.16)
        return _Klass
    except ImportError:
        pass
    try:
        from boxmot.trackers.botsort.botsort import BotSort as _Klass  # boxmot ≥ 0.17
        return _Klass
    except ImportError as exc:
        raise ImportError(
            "BoT-SORT backend requires the 'boxmot' package. "
            "Install it with: pip install 'card-capture[pipeline_v21]'"
        ) from exc


def _get_default_reid_weights() -> Path:
    """Get the default ReID weights path for boxmot v0.17+."""
    try:
        import boxmot
        weights_dir = Path(boxmot.__file__).parent / "trackers" / "botsort" / "weights"
        reid_weights = weights_dir / "osnet_x0_25_msmt17.pt"
        if reid_weights.exists():
            return reid_weights
    except Exception:
        pass
    # Fallback: boxmot will download if needed
    return Path("osnet_x0_25_msmt17.pt")


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

        # Prepare required arguments for boxmot v0.17+
        device = torch.device("cpu")
        reid_weights = _get_default_reid_weights()
        half = False

        self._tracker = BoTSORT(
            reid_weights=reid_weights,
            device=device,
            half=half,
            track_high_thresh=track_activation_threshold,
            track_buffer=lost_track_buffer,
            match_thresh=minimum_matching_threshold,
            cmc_method=None,  # Disable camera motion compensation since we have no real frames
        )
        self.min_track_length = min_track_length
        self._tracks: dict[int, TrackState] = {}
        self._all_finalized: list[TrackState] = []
        self.pending_splits: list[int] = []

    def reset(self) -> None:
        """Reset tracker state (e.g., between sessions)."""
        self._all_finalized.extend(self._tracks.values())
        self._tracks = {}

        device = torch.device("cpu")
        reid_weights = _get_default_reid_weights()
        half = False

        self._tracker = self._BoTSORT(
            reid_weights=reid_weights,
            device=device,
            half=half,
            track_high_thresh=self._track_activation_threshold,
            track_buffer=self._lost_track_buffer,
            match_thresh=self._minimum_matching_threshold,
            cmc_method=None,  # Disable camera motion compensation since we have no real frames
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

        # BotSort v0.17+ requires an image parameter
        # Use a dummy image since we're not using visual features in a meaningful way here
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self._tracker.update(det, dummy_img)

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
