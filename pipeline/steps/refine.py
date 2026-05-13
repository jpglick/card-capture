"""Step 4 — refine: Stage 6 GPU/CPU perspective rectification.

Decodes only the high-res frames needed for each track's canonical
candidates, runs Kornia GPU warp (falling back to the CPU cropper),
saves 750×1050 JPEG crops, and returns image paths + metadata.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.steps.start import RunContext
from pipeline.steps.track import TrackOutput


@dataclass
class RefineOutput:
    """Outputs of the refine step.

    ``refined_tracks`` is a list of dicts, one per finalized track, each
    containing the list of ``frame_entries`` (image paths + metadata)
    and the ``canonical_entries`` selected from them.
    """

    refined_tracks: List[Dict[str, Any]]
    tracks_data: List[Dict[str, Any]]  # forwarded verbatim
    detection_rows: List[Dict[str, Any]]
    sampler_telemetry: Dict[str, Any]
    bg_model_path: Optional[str]
    tracker_events: List[Dict[str, Any]]
    accepted_frame_presence: List[Tuple[int, int, bool]]
    frame_count: int
    accepted_frame_count: int
    video_id: int


def run(ctx: RunContext, track_out: TrackOutput) -> RefineOutput:
    """Decode high-res frames and warp each candidate to 750×1050.

    Args:
        ctx:       RunContext from the start step.
        track_out: Output from the track step.

    Returns:
        ``RefineOutput`` with per-track lists of rectified image paths.
    """
    import cv2
    import numpy as np
    from pathlib import Path as _Path

    from card_capture.cropper import PrecisionNormalizer
    from card_capture.gpu_refinement import KorniaNormalizer
    from card_capture.deduplicator import VisualDeduplicator
    from card_capture.fuser import find_glare_centroid
    from card_capture.scoring import QualityScorer
    from card_capture.presence.background_novelty import quad_novelty
    from card_capture.pipeline import _select_canonical_entries, _glare_mask, _laplacian_heatmap, _compress_array
    from card_capture.ingestion import _open_capture
    from card_capture.selector import ScoredCandidate
    from card_capture.models import QualityScore
    from card_capture.storage import Storage

    video_path = _Path(ctx.video_path)
    crops_dir = _Path(ctx.crops_dir)
    output_dir = _Path(ctx.output_dir)

    tracks_data = track_out.tracks_data
    detection_rows_list = track_out.detection_rows

    # Build lookup: detection_id → detection row dict
    detection_lookup: Dict[int, Dict[str, Any]] = {
        row["detection_id"]: row for row in detection_rows_list
    }

    # Determine which high-res frames to decode
    canonical_indices: set = set()
    for track_dict in tracks_data:
        for c in track_dict["candidates"]:
            if c["frame_index"] is not None:
                canonical_indices.add(int(c["frame_index"]))

    decoded_images: Dict[int, np.ndarray] = {}
    if canonical_indices:
        sampler_telemetry = track_out.sampler_telemetry
        valley_split_frames = set(sampler_telemetry.get("last_valley_splits") or [])
        capture = _open_capture(video_path)
        try:
            curr_idx = 0
            max_target = max(canonical_indices)
            while curr_idx <= max_target:
                ok, frame = capture.read()
                if not ok:
                    break
                if curr_idx in canonical_indices:
                    decoded_images[curr_idx] = frame
                curr_idx += 1
        finally:
            capture.release()

    # Set up Kornia normalizer
    normalizer = PrecisionNormalizer()
    kornia_normalizer: Optional[KorniaNormalizer] = None
    if ctx.use_kornia:
        try:
            kornia_normalizer = KorniaNormalizer(
                width=normalizer.width,
                height=normalizer.height,
                device=ctx.kornia_device,
            )
        except Exception:
            kornia_normalizer = None

    deduplicator = VisualDeduplicator()
    scorer = QualityScorer()
    storage = Storage(_Path(ctx.db_path))

    # Load background model if available
    bg_model = None
    if track_out.bg_model_path and _Path(track_out.bg_model_path).exists():
        from card_capture.presence.background_novelty import BackgroundModel
        mean_bgr = np.load(track_out.bg_model_path)
        bg_model = BackgroundModel.__new__(BackgroundModel)
        bg_model.mean_bgr = mean_bgr

    refined_tracks: List[Dict[str, Any]] = []
    t_refine_start = time.time()

    for track_dict in tracks_data:
        instance_id = track_dict["instance_id"]
        candidates_data = track_dict["candidates"]

        # Sort by score and take top 8 for canonical selection
        scored_candidates = sorted(candidates_data, key=lambda c: c["score_total"], reverse=True)[:8]

        # Batch Kornia warp if available
        normalized_by_detection: Dict[int, np.ndarray] = {}
        if kornia_normalizer is not None and scored_candidates:
            batch_items = []
            batch_ids = []
            for c in scored_candidates:
                raw = decoded_images.get(c["frame_index"])
                if raw is None:
                    h = detection_lookup.get(c["detection_id"], {}).get("height", 10)
                    w = detection_lookup.get(c["detection_id"], {}).get("width", 10)
                    raw = np.zeros((h, w, 3), dtype=np.uint8)
                batch_items.append((raw, c["corners"]))
                batch_ids.append(c["detection_id"])
            if batch_items:
                try:
                    warped = kornia_normalizer.warp_canonical_batch(batch_items, rotate_180=ctx.rotate_180)
                    for did, img in zip(batch_ids, warped):
                        normalized_by_detection[did] = img
                except Exception as e:
                    print(f"Kornia warp failed: {e}")
                    normalized_by_detection = {}

        frame_entries = []
        for c in scored_candidates:
            raw = decoded_images.get(c["frame_index"])
            det_row = detection_lookup.get(c["detection_id"], {})
            if raw is None:
                h = det_row.get("height", 10)
                w = det_row.get("width", 10)
                raw = np.zeros((h, w, 3), dtype=np.uint8)

            normalized = normalized_by_detection.get(c["detection_id"])
            if normalized is None:
                normalized = normalizer.normalize(raw, c["corners"], rotate_180=ctx.rotate_180)

            # Compute novelty score
            novelty_score = 1.0
            if bg_model is not None and c["corners"]:
                try:
                    novelty_score = float(quad_novelty(
                        raw, c["corners"], bg_model,
                        color_space="lab", lab_weights=(1.0, 0.5, 0.5),
                    ))
                except Exception:
                    pass

            quality_score = scorer.score(
                normalized,
                float(c["confidence"]),
                novelty=novelty_score,
            )
            glare_centroid = find_glare_centroid(normalized)
            glare_x, glare_y = glare_centroid if glare_centroid else (None, None)

            frame_entries.append({
                "candidate": c,         # original dict
                "normalized": normalized,  # np.ndarray (in-memory only)
                "quality_score": quality_score,
                "visual_hash": deduplicator.compute_phash(normalized),
                "glare_x": glare_x,
                "glare_y": glare_y,
                "sharpness": quality_score.components.get("sharpness", 0.0),
                "glare_mask": _compress_array(_glare_mask(normalized)),
                "laplacian_heatmap": _compress_array(_laplacian_heatmap(normalized)),
            })

        if not frame_entries:
            continue

        # Build ScoredCandidate-like objects for _select_canonical_entries
        from card_capture.selector import ScoredCandidate as _SC
        from card_capture.models import QualityScore as _QS

        def _to_scored(entry):
            c = entry["candidate"]
            qs = entry["quality_score"]
            return _SC(
                detection_id=c["detection_id"],
                timestamp_ms=c.get("timestamp_ms", 0),
                image_path=c.get("image_path", ""),
                score=qs,
                corners=[(float(x), float(y)) for x, y in c["corners"]] if c["corners"] else [],
                frame_index=c["frame_index"],
            )

        # Patch frame_entries to have "candidate" as a ScoredCandidate
        patched_entries = []
        for entry in frame_entries:
            sc = _to_scored(entry)
            patched_entries.append({**entry, "candidate": sc})

        canonical_entries = _select_canonical_entries(patched_entries, deduplicator)
        canonical_detection_ids = {e["candidate"].detection_id for e in canonical_entries}
        best_canonical = max(canonical_entries, key=lambda e: e["quality_score"].total)

        # Save rectified images to disk
        frame_entry_paths: List[Dict[str, Any]] = []
        for entry in frame_entries:
            c = entry["candidate"]
            det_id = c["detection_id"]
            is_canonical = det_id in canonical_detection_ids

            # Persist candidate image
            img_path = crops_dir / f"track_{instance_id[:8]}_det_{det_id}_rectified.jpg"
            cv2.imwrite(str(img_path), entry["normalized"])

            frame_entry_paths.append({
                "detection_id": det_id,
                "frame_index": c["frame_index"],
                "timestamp_ms": c.get("timestamp_ms", 0),
                "image_path": str(img_path),
                "quality_score": entry["quality_score"].total,
                "quality_components": dict(entry["quality_score"].components),
                "visual_hash": str(entry["visual_hash"]),
                "glare_x": entry["glare_x"],
                "glare_y": entry["glare_y"],
                "sharpness": entry["sharpness"],
                "is_canonical": is_canonical,
                "confidence": float(c.get("confidence", 0.0)),
                "corners": [(float(x), float(y)) for x, y in c["corners"]] if c["corners"] else [],
                "source_frame_path": c.get("image_path", ""),
                "triage_metrics": detection_lookup.get(det_id, {}).get("triage_metrics", {}),
                "width": detection_lookup.get(det_id, {}).get("width", 0),
                "height": detection_lookup.get(det_id, {}).get("height", 0),
                "novelty_score": detection_lookup.get(det_id, {}).get("novelty_score", 1.0),
            })

        # Save track telemetry for each canonical candidate
        for entry in canonical_entries:
            sc = entry["candidate"]
            if sc.corners:
                from card_capture.selector import _get_polygon_area, _aspect_ratio
                try:
                    area = _get_polygon_area(sc.corners)
                    aspect = _aspect_ratio(sc.corners)
                    cx = sum(p[0] for p in sc.corners) / 4.0
                    cy = sum(p[1] for p in sc.corners) / 4.0
                    storage.add_track_telemetry(
                        ctx.video_id, instance_id, sc.frame_index, area, aspect, cx, cy
                    )
                except Exception:
                    pass

        best_image_path = None
        for ep in frame_entry_paths:
            if ep["detection_id"] == best_canonical["candidate"].detection_id:
                best_image_path = ep["image_path"]
                break
        if best_image_path is None and frame_entry_paths:
            best_image_path = frame_entry_paths[0]["image_path"]

        refined_tracks.append({
            "instance_id": instance_id,
            "track_id": track_dict.get("track_id", 0),
            "angle": track_dict["angle"],
            "session_id": track_dict.get("session_id", 0),
            "first_frame_index": track_dict.get("first_frame_index", -1),
            "frame_entries": frame_entry_paths,
            "canonical_detection_ids": list(canonical_detection_ids),
            "best_canonical_detection_id": best_canonical["candidate"].detection_id,
            "best_canonical_image_path": best_image_path or "",
            "reid_embedding": track_dict.get("reid_embedding"),
        })

    t_refine = time.time() - t_refine_start
    print(f"[Stage: Refinement] | {t_refine:.2f}s | Refined {len(refined_tracks)} tracks")

    return RefineOutput(
        refined_tracks=refined_tracks,
        tracks_data=tracks_data,
        detection_rows=detection_rows_list,
        sampler_telemetry=track_out.sampler_telemetry,
        bg_model_path=track_out.bg_model_path,
        tracker_events=track_out.tracker_events,
        accepted_frame_presence=track_out.accepted_frame_presence,
        frame_count=track_out.frame_count,
        accepted_frame_count=track_out.accepted_frame_count,
        video_id=track_out.video_id,
    )
