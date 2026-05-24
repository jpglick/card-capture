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
    from card_capture.pipeline_utils import (
        _select_canonical_entries, _glare_mask, _compress_array,
        decode_frames_gpu, _compute_laplacian_scan_indices,
    )
    from card_capture.selector import ScoredCandidate
    from card_capture.models import QualityScore
    from card_capture.storage import Storage
    from card_capture.ml.models.dino_embedder import DinoEmbedder

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

    # --- In-track Laplacian quality scan ---
    # Temporal stride selected frames at ~3fps per track. Within each confirmed
    # track's time window we scan densely (laplacian_scan_stride source frames)
    # to find sharper frames that the sparse YOLO pass may have skipped.
    # This runs as ONE forward video pass across all tracks — no repeated seeks.
    # Non-YOLO frames use corners borrowed from the nearest YOLO detection
    # (corners are stable over short holds; max_corner_gap_frames limits drift).
    from card_capture.pipeline_utils import _laplacian_select_frames
    _lap_top_k = max(1, ctx.fusion_target_frames)
    _lap_ranges = []
    for _td in tracks_data:
        _dets = [
            (int(c["frame_index"]), c["corners"])
            for c in _td["candidates"]
            if c["frame_index"] is not None
        ]
        if _dets:
            _lap_ranges.append({"instance_id": _td["instance_id"], "detections": _dets})

    # Compute union of all frame indices needed by Laplacian scan and Kornia warp,
    # then decode once via NVDEC instead of two separate CPU VideoCapture passes.
    _lap_scan_indices = _compute_laplacian_scan_indices(_lap_ranges, ctx.laplacian_scan_stride)
    _all_needed = canonical_indices | _lap_scan_indices
    decoded_images: Dict[int, np.ndarray] = {}
    if _all_needed:
        decoded_images = decode_frames_gpu(video_path, sorted(_all_needed))

    _lap_results: Dict[str, list] = {}
    try:
        _lap_results = _laplacian_select_frames(
            video_path,
            _lap_ranges,
            scan_stride=ctx.laplacian_scan_stride,
            top_k=_lap_top_k,
            max_corner_gap=ctx.max_corner_gap_frames,
            decoded_frames=decoded_images if decoded_images else None,
        )
    except Exception as _e:
        print(f"[Refine] Laplacian scan failed, falling back to temporal-stride frames: {_e}")

    # Add any non-YOLO Laplacian-selected frames to the canonical decode set
    for _sel_list in _lap_results.values():
        for _fi, _ in _sel_list:
            canonical_indices.add(int(_fi))
    # ----------------------------------------

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
    
    # Set up DINOv2 embedder (ONCE per video run)
    try:
        embedder = DinoEmbedder(variant="vits14")
    except Exception as e:
        print(f"Failed to load DINOv2 embedder: {e}")
        embedder = None

    # Load background model if available
    bg_model = None
    if track_out.bg_model_path and _Path(track_out.bg_model_path).exists():
        from card_capture.presence.background_novelty import BackgroundModel
        mean_bgr = np.load(track_out.bg_model_path)
        bg_model = BackgroundModel.__new__(BackgroundModel)
        bg_model.mean_bgr = mean_bgr

    refined_tracks: List[Dict[str, Any]] = []
    t_refine_start = time.time()

    # Per-op timing aggregation across all tracks/candidates so we can identify
    # which CPU-bound steps dominate the refine step's wall time. Surfaces in
    # stage_payloads.stage_refine in the next handler diagnostic dump.
    _t_ops: Dict[str, float] = {
        "decode_setup": 0.0,
        "corner_refinement": 0.0,
        "kornia_warp_batch": 0.0,
        "precision_normalizer_fallback": 0.0,
        "quality_scoring": 0.0,
        "novelty_scoring": 0.0,
        "glare_centroid": 0.0,
        "glare_mask": 0.0,
        "phash": 0.0,
        "imwrite": 0.0,
        "reid_embed_batch": 0.0,
        "other": 0.0,
    }
    _op_counts: Dict[str, int] = {k: 0 for k in _t_ops}

    def _tick(label: str, started_at: float) -> float:
        _t_ops[label] += time.time() - started_at
        _op_counts[label] += 1
        return time.time()

    for track_dict in tracks_data:
        instance_id = track_dict["instance_id"]
        candidates_data = track_dict["candidates"]

        # Sort by score and take top 8 for canonical selection
        scored_candidates = sorted(candidates_data, key=lambda c: c["score_total"], reverse=True)[:8]

        # Reorder: put Laplacian-selected frames first. This overrides the
        # quality-score ranking so the sharpest frame(s) from the dense scan
        # become the canonical output. Frames not in the YOLO detection set
        # (picked from between detections) are inserted as synthetic candidates
        # with borrowed corners; the warp logic treats them identically.
        _lap_frames = _lap_results.get(instance_id, [])
        if _lap_frames:
            _existing_fi = {c.get("frame_index") for c in candidates_data}
            _source_fps = track_out.sampler_telemetry.get("last_source_fps", 30.0) or 30.0
            _lap_leading = []
            _lap_fi_set = set()
            for _fi, _corners in _lap_frames:
                _lap_fi_set.add(_fi)
                if _fi in _existing_fi:
                    # Already a YOLO detection — pull it to front with its own data
                    _match = next((c for c in candidates_data if c.get("frame_index") == _fi), None)
                    if _match:
                        _lap_leading.append(_match)
                else:
                    # Non-YOLO frame: synthesize a minimal candidate using
                    # borrowed corners. score_total is a placeholder — ordering
                    # here is by Laplacian sharpness, not by this score.
                    _ref = candidates_data[0] if candidates_data else {}
                    _lap_leading.append({
                        "frame_index": _fi,
                        "corners": _corners,
                        "score_total": float(np.median([c["score_total"] for c in candidates_data])) if candidates_data else 0.7,
                        "detection_id": -1,
                        "confidence": float(np.median([c["confidence"] for c in candidates_data])) if candidates_data else 0.7,
                        "width": _ref.get("width", 3840),
                        "height": _ref.get("height", 2160),
                        "timestamp_ms": int(_fi * 1000 / _source_fps),
                    })
            _remaining = [c for c in scored_candidates if c.get("frame_index") not in _lap_fi_set]
            scored_candidates = (_lap_leading + _remaining)[:8]

        # Apply RANSAC corner refinement if enabled
        if ctx.corner_refinement:
            _t = time.time()
            from card_capture.ml.inference.corner_refinement import refine_corners
            for c in scored_candidates:
                raw = decoded_images.get(c["frame_index"])
                if raw is not None and c["corners"]:
                    try:
                        c["corners"] = refine_corners(raw, c["corners"])
                    except Exception as e:
                        print(f"Corner refinement failed for {instance_id}: {e}")
            _tick("corner_refinement", _t)

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
                    _t = time.time()
                    warped = kornia_normalizer.warp_canonical_batch(batch_items, rotate_180=ctx.rotate_180)
                    _tick("kornia_warp_batch", _t)
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
                _t = time.time()
                normalized = normalizer.normalize(raw, c["corners"], rotate_180=ctx.rotate_180)
                _tick("precision_normalizer_fallback", _t)

            # Compute novelty score
            novelty_score = 1.0
            if bg_model is not None and c["corners"]:
                try:
                    _t = time.time()
                    novelty_score = float(quad_novelty(
                        raw, c["corners"], bg_model,
                        color_space="lab", lab_weights=(1.0, 0.5, 0.5),
                    ))
                    _tick("novelty_scoring", _t)
                except Exception:
                    pass

            _t = time.time()
            quality_score = scorer.score(
                normalized,
                float(c["confidence"]),
                novelty=novelty_score,
            )
            _tick("quality_scoring", _t)

            _t = time.time()
            glare_centroid = find_glare_centroid(normalized)
            glare_x, glare_y = glare_centroid if glare_centroid else (None, None)
            _tick("glare_centroid", _t)

            _t = time.time()
            _phash = deduplicator.compute_phash(normalized)
            _tick("phash", _t)

            _t = time.time()
            _gmask = _compress_array(_glare_mask(normalized))
            _tick("glare_mask", _t)

            frame_entries.append({
                "candidate": c,
                "normalized": normalized,
                "quality_score": quality_score,
                "visual_hash": _phash,
                "glare_x": glare_x,
                "glare_y": glare_y,
                "sharpness": quality_score.components.get("sharpness", 0.0),
                "glare_mask": _gmask,
            })

        if not frame_entries:
            continue
            
        # Build ScoredCandidate-like objects for _select_canonical_entries
        from card_capture.selector import ScoredCandidate as _SC

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
            _t = time.time()
            cv2.imwrite(str(img_path), entry["normalized"])
            _tick("imwrite", _t)

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
            "reid_embedding": None,  # filled in by batch embedding pass below
        })

    # Batch ReID embeddings for all tracks in one GPU pass
    if embedder:
        _embed_jobs = [
            (i, rt["best_canonical_image_path"])
            for i, rt in enumerate(refined_tracks)
            if rt["best_canonical_image_path"]
        ]
        if _embed_jobs:
            try:
                from PIL import Image as _PIL_Image
                _pil_images = [_PIL_Image.open(p).convert("RGB") for _, p in _embed_jobs]
                _t = time.time()
                _emb_tensor = embedder.embed_batch(_pil_images)
                _emb_np = _emb_tensor.cpu().numpy()
                _tick("reid_embed_batch", _t)
                for _batch_i, (_track_i, _) in enumerate(_embed_jobs):
                    refined_tracks[_track_i]["reid_embedding"] = _emb_np[_batch_i].tolist()
            except Exception as _e:
                print(f"[Refine] Batch embedding failed: {_e}")

    t_refine = time.time() - t_refine_start
    # Print the per-op breakdown so logs surface it even before stage_payloads
    # is queried. Sort by time descending so the biggest offenders are at top.
    print(f"[Stage: Refinement] | {t_refine:.2f}s | Refined {len(refined_tracks)} tracks")
    _sorted = sorted(_t_ops.items(), key=lambda kv: kv[1], reverse=True)
    for _label, _elapsed in _sorted:
        if _elapsed > 0.01:
            _calls = _op_counts[_label]
            _per = (_elapsed / _calls * 1000) if _calls else 0.0
            print(f"  refine_op: {_label:32s} {_elapsed:6.2f}s  ({_calls} calls, {_per:.1f}ms avg)")

    # Attach per-op timings to the RefineOutput so card_capture_flow's
    # _record_stage_timing can merge them into stage_detect-style telemetry.
    refine_telemetry = {
        "tracks_refined": len(refined_tracks),
        "op_seconds": {k: round(v, 3) for k, v in _t_ops.items() if v > 0},
        "op_calls": {k: _op_counts[k] for k in _t_ops if _op_counts[k] > 0},
    }

    out = RefineOutput(
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
    # Stash telemetry as an attribute so card_capture_flow can read it. Using
    # setattr (rather than expanding the dataclass) keeps the change minimal
    # and doesn't break any consumer that pickles RefineOutput.
    out.refine_telemetry = refine_telemetry  # type: ignore[attr-defined]
    return out
