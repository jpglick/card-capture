from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..models import ScoredCandidate, TrackState, FramePacket
from .bytetrack_adapter import _AdaptedDetection, _xyxy_from_corners


def rotated_iou(corners_a: List[Tuple[float, float]], corners_b: List[Tuple[float, float]]) -> float:
    """
    Compute intersection-over-union (IoU) for oriented bounding boxes.

    Uses Shapely Polygon intersection when available; falls back to axis-aligned IoU
    if Shapely fails or is unavailable.

    Args:
        corners_a: List of 4 (x, y) tuples representing OBB corners of first box
        corners_b: List of 4 (x, y) tuples representing OBB corners of second box

    Returns:
        IoU value in range [0, 1]
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        # Fallback: use axis-aligned IoU if Shapely not available
        return axis_aligned_iou_from_corners(corners_a, corners_b)

    try:
        # Convert corner lists to Polygons
        poly_a = Polygon(corners_a)
        poly_b = Polygon(corners_b)

        # Compute intersection and union areas
        intersection = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area

        # Return IoU
        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception:
        # Fallback to axis-aligned IoU if Shapely fails
        return axis_aligned_iou_from_corners(corners_a, corners_b)


def axis_aligned_iou_from_corners(corners_a: List[Tuple[float, float]], corners_b: List[Tuple[float, float]]) -> float:
    """
    Compute axis-aligned IoU from corner coordinates.

    Extracts axis-aligned bounding boxes from corners and computes standard IoU.

    Args:
        corners_a: List of 4 (x, y) tuples
        corners_b: List of 4 (x, y) tuples

    Returns:
        IoU value in range [0, 1]
    """
    # Extract axis-aligned bboxes from corners
    def bbox_from_corners(corners):
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    x1a, y1a, x2a, y2a = bbox_from_corners(corners_a)
    x1b, y1b, x2b, y2b = bbox_from_corners(corners_b)

    # Compute intersection box
    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)

    # Intersection area
    intersection = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)

    # Union area
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union = area_a + area_b - intersection

    # IoU
    if union == 0:
        return 0.0
    return float(intersection / union)


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
            track_low_thresh=max(0.1, track_activation_threshold - 0.20),
            new_track_thresh=track_activation_threshold,
            track_buffer=lost_track_buffer,
            match_thresh=minimum_matching_threshold,
            cmc_method=None,  # Disable camera motion compensation (assumes static camera setup)
            # The pipeline applies min_track_length after sessionization; keep
            # BoT-SORT from hiding short but valid sessions before that gate.
            min_hits=1,
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
            track_low_thresh=max(0.1, self._track_activation_threshold - 0.20),
            new_track_thresh=self._track_activation_threshold,
            track_buffer=self._lost_track_buffer,
            match_thresh=self._minimum_matching_threshold,
            cmc_method=None,  # Disable camera motion compensation (assumes static camera setup)
            # The pipeline applies min_track_length after sessionization; keep
            # BoT-SORT from hiding short but valid sessions before that gate.
            min_hits=1,
        )
        self.pending_splits = []

    def finalized_tracks(self) -> List[TrackState]:
        return list(self._all_finalized)

    def _decode_frame_for_reid(self, candidates: List[ScoredCandidate]) -> np.ndarray:
        """
        Attempt to decode the source frame from the first candidate's image_path.

        If successful, returns the decoded BGR frame for ReID feature extraction.
        If decoding fails or no path available, falls back to a zero-filled dummy image.

        Args:
            candidates: List of ScoredCandidate objects for the current frame.

        Returns:
            Decoded frame (BGR, uint8) or fallback zeros array of shape (480, 640, 3).
        """
        import cv2

        # Try to use the first candidate's image_path
        if candidates and candidates[0].image_path:
            try:
                frame = cv2.imread(candidates[0].image_path)
                if frame is not None:
                    return frame
            except Exception:
                # Decode failed; fall back to zeros
                pass

        # Fallback: dummy image (current behavior)
        return np.zeros((480, 640, 3), dtype=np.uint8)

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

        # boxmot's trackers expect a numpy array of shape (N, 6)
        # Format: [x1, y1, x2, y2, conf, class_id]
        dets_array = []
        for b, c in zip(boxes, confidences):
            dets_array.append([b[0], b[1], b[2], b[3], c, 0])
        det_input = np.array(dets_array, dtype=np.float32)

        # BotSort v0.17+ requires an image parameter.
        # Try to decode real frame from path reference; fall back to zeros if not available.
        frame_img = self._decode_frame_for_reid(candidates)
        
        # boxmot's trackers return an ndarray of shape (N, 8) 
        # [x1, y1, x2, y2, track_id, conf, class_id, ...]
        tracks = self._tracker.update(det_input, frame_img)
        # print(f"DEBUG: tracker.update returned {len(tracks)} tracks. input was {len(det_input)}")

        out: List[_AdaptedDetection] = []
        if tracks is None or len(tracks) == 0:
            return out

        for track in tracks:
            tid = int(track[4])
            if tid == -1:
                continue
                
            # Match back to candidate using IoU since detections might be re-ordered
            box = track[:4]
            best_cand = None
            best_iou = -1
            for cand in valid_candidates:
                xs = [p[0] for p in cand.corners]
                ys = [p[1] for p in cand.corners]
                cx1, cy1 = min(xs), min(ys)
                cx2, cy2 = max(xs), max(ys)
                
                ix1, iy1 = max(box[0], cx1), max(box[1], cy1)
                ix2, iy2 = min(box[2], cx2), min(box[3], cy2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                if iw > 0 and ih > 0:
                    inter = iw * ih
                    uni = (box[2]-box[0])*(box[3]-box[1]) + (cx2-cx1)*(cy2-cy1) - inter
                    iou = inter / uni
                    if iou > best_iou:
                        best_iou = iou
                        best_cand = cand
                        
            if best_cand is None:
                continue
                
            if tid not in self._tracks:
                self._tracks[tid] = TrackState(
                    instance_id=str(uuid.uuid4()),
                    candidates=[],
                    last_centroid=None,
                    last_frame_index=best_cand.frame_index,
                )
            state = self._tracks[tid]
            
            # Extract ReID embedding from the internal tracker's track objects
            # We look for the track object with matching ID in active_tracks
            internal_track = None
            for t in self._tracker.active_tracks:
                if t.track_id == tid:
                    internal_track = t
                    break
            
            if internal_track is not None:
                # Use smooth_feat (exponential moving average) for more stable identity
                if hasattr(internal_track, "smooth_feat") and internal_track.smooth_feat is not None:
                    state.reid_embedding = np.array(internal_track.smooth_feat, copy=True)
                elif hasattr(internal_track, "curr_feat") and internal_track.curr_feat is not None:
                    state.reid_embedding = np.array(internal_track.curr_feat, copy=True)

            state.candidates.append(best_cand)
            state.last_frame_index = best_cand.frame_index
            out.append(_AdaptedDetection(candidate=best_cand, track_id=tid, instance_id=state.instance_id))

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
