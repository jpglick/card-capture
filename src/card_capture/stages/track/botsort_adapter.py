from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from card_capture.core.models import ScoredCandidate, TrackState, FramePacket
from card_capture.pipeline.stages.dedup import SAME_CARD_EMB_THRESHOLD
from .bytetrack_adapter import _AdaptedDetection, _xyxy_from_corners
from .centroid_jump import CentroidJumpDetector
from .appearance_sessionizer import AppearanceObservation, AppearanceSessionizer


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
    except Exception as e1:
        e_first = e1
    try:
        from boxmot import BotSort as _Klass  # boxmot ≥ 0.17 top-level export
        return _Klass
    except Exception as e2:
        e_second = e2
    try:
        from boxmot.trackers.botsort.botsort import BotSort as _Klass  # boxmot ≥ 0.17
        return _Klass
    except Exception as exc:
        raise ImportError(
            f"BoT-SORT backend requires the 'boxmot' package. "
            f"Install it with: pip install 'card-capture[legacy_tracking]'. "
            f"Errors: {e_first}, {e_second}, {exc}"
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


def _get_shared_embedder():
    from card_capture.pipeline.stages.refine import get_shared_embedder

    return get_shared_embedder()


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
        centroid_jump_ratio: float = 0.30,
        centroid_jump_frames: int = 3,
        **appearance_kwargs,
    ) -> None:
        BoTSORT = _import_botsort()
        self._BoTSORT = BoTSORT
        self._track_activation_threshold = track_activation_threshold
        self._lost_track_buffer = lost_track_buffer
        self._minimum_matching_threshold = minimum_matching_threshold
        self.centroid_jump_ratio = centroid_jump_ratio
        self.centroid_jump_frames = centroid_jump_frames
        self._session_id = 0
        self.last_reset_count = 0
        self.centroid_jump_count = 0
        self._centroid_detector = CentroidJumpDetector(
            jump_ratio=centroid_jump_ratio,
            jump_within_frames=centroid_jump_frames,
        )
        self._sessionizer = AppearanceSessionizer(**appearance_kwargs)
        self.sessionization_metrics: dict[str, object] = {}

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
            appearance_thresh=SAME_CARD_EMB_THRESHOLD,
            with_reid=True,
            # The pipeline applies min_track_length after sessionization; keep
            # BoT-SORT from hiding short but valid sessions before that gate.
            min_hits=1,
        )
        self.min_track_length = min_track_length
        self._tracks: dict[int, TrackState] = {}
        self._all_finalized: list[TrackState] = []
        self.pending_splits: list[int] = []

    def reset(self, *, count_metric: bool = True) -> None:
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
            appearance_thresh=SAME_CARD_EMB_THRESHOLD,
            with_reid=True,
            # The pipeline applies min_track_length after sessionization; keep
            # BoT-SORT from hiding short but valid sessions before that gate.
            min_hits=1,
        )
        self.pending_splits = []
        self._centroid_detector.reset()
        if count_metric:
            self.last_reset_count += 1

    def finalized_tracks(self) -> List[TrackState]:
        return list(self._all_finalized)

    def _decode_frame_for_reid(
        self,
        candidates: List[ScoredCandidate],
        frame_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Attempt to decode the source frame from the first candidate's image_path.

        If successful, returns the decoded BGR frame for ReID feature extraction.
        If decoding fails or no path available, falls back to a zero-filled dummy image.

        Args:
            candidates: List of ScoredCandidate objects for the current frame.

        Returns:
            Decoded frame (BGR, uint8) or fallback zeros array of shape (480, 640, 3).
        """
        if frame_img is not None:
            return frame_img

        if candidates and candidates[0].image_path:
            try:
                from PIL import Image

                with Image.open(candidates[0].image_path) as img:
                    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
                return rgb[..., ::-1]
            except Exception:
                pass

        # Fallback: dummy image (current behavior)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _embed_candidates(
        self,
        candidates: List[ScoredCandidate],
        frame_img: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if frame_img is None:
            return None
        embedder = _get_shared_embedder()
        if embedder is None:
            return None

        dim = int(getattr(embedder, "dim", 0) or 0)
        rows: list[np.ndarray] = []
        h, w = frame_img.shape[:2]
        
        for cand in candidates:
            emb = None
            if cand.corners:
                x1f, y1f, x2f, y2f = _xyxy_from_corners(cand.corners)
                x1, y1 = max(0, int(np.floor(x1f))), max(0, int(np.floor(y1f)))
                x2, y2 = min(w, int(np.ceil(x2f))), min(h, int(np.ceil(y2f)))
                
                if x2 > x1 and y2 > y1:
                    crop_bgr = frame_img[y1:y2, x1:x2]
                    if crop_bgr.size > 0:
                        crop_rgb = crop_bgr[..., ::-1]
                        res = embedder.embed_array(crop_rgb)
                        if res is not None:
                            emb = np.asarray(res.cpu().numpy(), dtype=np.float32).reshape(-1)
                            if dim <= 0 and emb.size > 0:
                                dim = int(emb.size)

            if emb is None or emb.size == 0:
                # Use a deferred zero-fill if dim is still unknown
                rows.append(None)  # type: ignore
            else:
                rows.append(emb)

        if not rows:
            return None
            
        # Fill in any missing embeddings with zeros once we know the dimension
        if dim <= 0:
            # If we still don't know the dim, we can't return a valid array
            return None
            
        final_rows = []
        for r in rows:
            if r is None or r.size != dim:
                final_rows.append(np.zeros(dim, dtype=np.float32))
            else:
                final_rows.append(r)
                
        return np.asarray(final_rows, dtype=np.float32)

    def process(
        self,
        candidates: List[ScoredCandidate],
        frame_img: Optional[np.ndarray] = None,
        embs: Optional[np.ndarray] = None,
    ) -> List[_AdaptedDetection]:
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
        had_real_frame = frame_img is not None
        frame_img = self._decode_frame_for_reid(candidates, frame_img=frame_img)
        if embs is None and had_real_frame:
            embs = self._embed_candidates(valid_candidates, frame_img=frame_img)
        if embs is not None and len(embs) != len(valid_candidates):
            embs = None
        
        # boxmot's trackers return an ndarray of shape (N, 8) 
        # [x1, y1, x2, y2, track_id, conf, class_id, ...]
        try:
            tracks = self._tracker.update(det_input, frame_img, embs=embs)
        except TypeError:
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
                    session_id=self._session_id,
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
        
        # Merge tracks with the same session_id. AppearanceSessionizer is the
        # ultimate authority on session boundaries; if BoTSORT fragmented a track
        # due to proximity/IoU loss within a stable appearance plateau, we heal it here.
        session_to_track: dict[int, TrackState] = {}
        for t in all_tracks:
            sid = getattr(t, "session_id", None)
            if sid is None:
                # If no session_id (e.g. from tests), just keep it distinct
                session_to_track[id(t)] = t
                continue
            
            if sid not in session_to_track:
                session_to_track[sid] = t
            else:
                existing = session_to_track[sid]
                existing.candidates.extend(t.candidates)
                existing.candidates.sort(key=lambda c: c.frame_index)
                if getattr(t, "last_frame_index", 0) > getattr(existing, "last_frame_index", 0):
                    existing.last_frame_index = t.last_frame_index
                if getattr(t, "reid_embedding", None) is not None:
                    existing.reid_embedding = getattr(t, "reid_embedding", None)

        merged_tracks = list(session_to_track.values())
        return [t for t in merged_tracks if len(t.candidates) >= self.min_track_length]

    def assign(
        self,
        detections: List[dict] | List[ScoredCandidate],
        frames: List[dict] | List[FramePacket],
    ) -> List[TrackState]:
        """Unified entry point for the pipeline 'track' stage.

        Processes all detections frame-by-frame and returns finalized tracks.
        """
        self._session_id = 0
        self.last_reset_count = 0
        self.centroid_jump_count = 0
        self.reset(count_metric=False)
        if not detections:
            return []

        # Group detections by frame_index
        by_frame: dict[int, list[ScoredCandidate]] = {}
        frame_width_by_index: dict[int, int] = {}
        # Per-detection novelty lives on the source dicts, not on
        # ScoredCandidate, so capture it here for the sessionizer's bridge gate.
        novelty_by_det_id: dict = {}
        for d in detections:
            if isinstance(d, dict):
                # Convert dict to ScoredCandidate if needed
                from card_capture.core.models import QualityScore, ScoredCandidate
                # Use detection confidence if available, else novelty_score
                total_score = float(d.get("confidence", d.get("novelty_score", 1.0)))
                cand = ScoredCandidate(
                    detection_id=d.get("detection_id", str(uuid.uuid4())),
                    timestamp_ms=d.get("timestamp_ms", 0),
                    image_path=d.get("image_path", ""),
                    score=QualityScore(total=total_score, components={}),
                    corners=d.get("corners", []),
                    frame_index=d.get("frame_index", 0),
                )
                frame_idx = int(d.get("frame_index", 0))
                width = int(d.get("width", 0) or 0)
                if width > 0 and frame_idx not in frame_width_by_index:
                    frame_width_by_index[frame_idx] = width
                novelty_by_det_id[cand.detection_id] = float(d.get("novelty_score", 1.0))
            else:
                cand = d
            
            by_frame.setdefault(cand.frame_index, []).append(cand)

        # Temporary mapping for Step 1, avoid storing all in a long-lived dict if possible
        frame_img_map: dict[int, np.ndarray] = {}
        for frame in frames:
            if isinstance(frame, dict):
                idx = int(frame.get("frame_index", -1))
                img = frame.get("image")
            else:
                idx = int(getattr(frame, "frame_index", -1))
                img = getattr(frame, "image", None)
            if idx >= 0 and isinstance(img, np.ndarray):
                frame_img_map[idx] = img
                if idx not in frame_width_by_index:
                    frame_width_by_index[idx] = int(img.shape[1])

        # Step 1: Precompute embeddings and collect observations
        observations: list[AppearanceObservation] = []
        frame_data: dict[int, tuple[list[ScoredCandidate], Optional[np.ndarray], Optional[np.ndarray]]] = {}
        
        for idx in sorted(by_frame.keys()):
            frame_candidates = by_frame[idx]
            frame_img = frame_img_map.get(idx)
            frame_width = int(frame_width_by_index.get(idx, 640))
            
            # Update centroid telemetry
            best_candidate = max(
                (c for c in frame_candidates if c.corners),
                key=lambda c: float(c.score.total),
                default=None,
            )
            if best_candidate is not None and best_candidate.corners:
                corners = np.asarray(best_candidate.corners, dtype=np.float32)
                if self._centroid_detector.update(corners, frame_width=max(1, frame_width)):
                    self.centroid_jump_count += 1
            
            # Filter valid candidates for embedding to ensure consistency with process()
            valid_for_embs = [c for c in frame_candidates if c.corners]
            embs = self._embed_candidates(valid_for_embs, frame_img=frame_img)
            
            # Memory optimization: avoid holding large arrays if we can decode them later.
            # If any candidate has a valid image_path, we can fallback to on-demand decoding.
            can_decode = any(bool(getattr(c, "image_path", "")) for c in valid_for_embs)
            stored_img = frame_img if not can_decode else None
            
            frame_data[idx] = (frame_candidates, stored_img, embs)
            
            if best_candidate is not None and embs is not None:
                try:
                    # Map best_candidate to its embedding in the filtered list
                    cand_idx = valid_for_embs.index(best_candidate)
                    best_emb = embs[cand_idx]
                    observations.append(AppearanceObservation(
                        frame_index=idx,
                        detection_id=best_candidate.detection_id,
                        embedding=best_emb,
                        novelty_score=float(
                            novelty_by_det_id.get(best_candidate.detection_id, 1.0)
                        ),
                    ))
                except (ValueError, IndexError):
                    pass
        
        # Clear temporary image map to free memory before sessionization and second pass
        frame_img_map.clear()

        # Step 2: Sessionize
        res = self._sessionizer.sessionize(observations)
        self.sessionization_metrics = res.metrics()
        self.sessionization_metrics["centroid_jump_count"] = self.centroid_jump_count
        
        # Step 3: Process retained plateaus
        if not res.retained_plateaus:
            # Fail open: process all frames in one session if no plateau is retained
            for idx in sorted(frame_data.keys()):
                frame_candidates, frame_img, embs = frame_data[idx]
                self.process(frame_candidates, frame_img=frame_img, embs=embs)
            return self.finalize()

        for i, plateau in enumerate(res.retained_plateaus):
            if i > 0:
                self.reset()
                self._session_id += 1
            
            for obs in plateau.observations:
                idx = obs.frame_index
                frame_candidates, frame_img, embs = frame_data[idx]
                self.process(frame_candidates, frame_img=frame_img, embs=embs)

        return self.finalize()
