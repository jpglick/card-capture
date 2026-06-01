"""Stage 6: GPU Refinement (Kornia perspective warp -> 750x1050).

V5.5 in-memory port of V4 ``pipeline/steps/refine.py``. Reads frames
from ``state["sampled_frames"]`` (never re-decodes), warps each track's
top candidates to 750x1050, attaches a QualityScore, a pHash, and glare
telemetry per frame_entry, picks a canonical set, and stashes the
best-canonical image in memory for downstream stages.

Three V4-vs-V5.5 substitutions (audited in P13):
- ``decoded_images[frame_index]`` -> ``state["sampled_frames"]`` lookup
- ``cv2.imwrite(... rectified.jpg)`` -> stored in ``frame_entry["normalized"]``
- ``embedder.embed_image(path)`` -> ``embedder.embed_array(img)``
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from card_capture.stages.refine.cropper import PrecisionNormalizer
from card_capture.deduplicator import VisualDeduplicator
from card_capture.stages.fuse.fuser import find_glare_centroid
from card_capture.stages.refine.gpu_refinement import KorniaNormalizer
from card_capture.core.models import FrameSample
from card_capture.shared.stage_metrics import emit_stage_metrics
from card_capture.shared.pipeline_utils import (
    _compress_array,
    _glare_mask,
    _laplacian_heatmap,
    _select_canonical_entries,
)
from card_capture.stages.score.scoring import QualityScorer
from card_capture.stages.score.selector import ScoredCandidate


_EMBEDDER_SINGLETON: object = None


def get_shared_embedder():
    """Return a DinoEmbedder singleton, or None if weights are missing.

    Cached at module scope so we don't reload the model per pipeline run."""
    global _EMBEDDER_SINGLETON
    if _EMBEDDER_SINGLETON is not None:
        return _EMBEDDER_SINGLETON
    try:
        from card_capture.ml.models.dino_embedder import DinoEmbedder
        _EMBEDDER_SINGLETON = DinoEmbedder(variant="vits14")
    except Exception:
        _EMBEDDER_SINGLETON = None
    return _EMBEDDER_SINGLETON


def _get_embedder():
    # Backwards-compatible alias for existing tests/patches.
    return get_shared_embedder()


def _frame_index_lookup(frames) -> Dict[int, np.ndarray]:
    return {int(f.frame_index): f.image for f in frames}


def _scored_candidate_from_dict(c: dict) -> ScoredCandidate:
    from card_capture.core.models import QualityScore
    return ScoredCandidate(
        detection_id=int(c["detection_id"]),
        timestamp_ms=int(c.get("timestamp_ms", 0)),
        image_path="",
        score=QualityScore(total=float(c.get("score_total", 0.0)), components={}),
        corners=[(float(x), float(y)) for x, y in (c.get("corners") or [])],
        frame_index=int(c["frame_index"]),
    )


def run(state: dict, *, telemetry) -> None:
    frames = state.get("sampled_frames")
    if frames is None:
        telemetry.contract_violation(
            "refine_without_frames",
            {"hint": "sample stage must populate state['sampled_frames']"},
        )
        raise RuntimeError("refine reached without sampled_frames in state")

    config = state["request"].config
    decoded_images = _frame_index_lookup(frames)

    normalizer = PrecisionNormalizer()
    kornia_normalizer: Optional[KorniaNormalizer] = None
    if config.get("use_kornia", True):
        try:
            kornia_normalizer = KorniaNormalizer(
                width=normalizer.width,
                height=normalizer.height,
                device=config.get("kornia_device", config.get("device", "auto")),
            )
        except Exception:
            kornia_normalizer = None

    deduplicator = VisualDeduplicator()
    scorer = QualityScorer()
    rotate_180 = bool(config.get("rotate_180", False))

    refined_tracks: List[Dict[str, Any]] = []
    tracks_data = state.get("tracks_data") or []
    
    total_tracks = len(tracks_data)

    for i, track_dict in enumerate(tracks_data):
        instance_id = track_dict["instance_id"]
        candidates_data = track_dict["candidates"]

        # Sort by score and take top 8 (matches V4 line 173)
        scored_candidates = sorted(
            candidates_data, key=lambda c: c.get("score_total", 0.0), reverse=True
        )[:8]

        # Batched Kornia warp for this track's candidates
        normalized_by_det: Dict[int, np.ndarray] = {}
        if kornia_normalizer is not None and scored_candidates:
            batch_items = []
            batch_ids = []
            for c in scored_candidates:
                raw = decoded_images.get(int(c["frame_index"]))
                if raw is None:
                    h = int(c.get("height", 10))
                    w = int(c.get("width", 10))
                    raw = np.zeros((h, w, 3), dtype=np.uint8)
                batch_items.append((raw, c["corners"]))
                batch_ids.append(int(c["detection_id"]))
            try:
                warped = kornia_normalizer.warp_canonical_batch(
                    batch_items, rotate_180=rotate_180
                )
                for did, img in zip(batch_ids, warped):
                    normalized_by_det[did] = img
            except Exception as exc:
                telemetry.resource_sample(
                    {"event": "kornia_warp_failed", "error": repr(exc)}
                )

        frame_entries: List[Dict[str, Any]] = []
        for c in scored_candidates:
            raw = decoded_images.get(int(c["frame_index"]))
            if raw is None:
                raw = np.zeros((int(c.get("height", 10)), int(c.get("width", 10)), 3),
                               dtype=np.uint8)
            normalized = normalized_by_det.get(int(c["detection_id"]))
            if normalized is None:
                normalized = normalizer.normalize(raw, c["corners"], rotate_180=rotate_180)

            quality_score = scorer.score(
                normalized,
                float(c.get("confidence", 0.0)),
                novelty=float(c.get("novelty_score", 1.0)),
            )
            glare_centroid = find_glare_centroid(normalized)
            glare_x, glare_y = glare_centroid if glare_centroid else (None, None)

            frame_entries.append({
                "candidate_dict": c,
                "candidate": _scored_candidate_from_dict({**c, "score_total": quality_score.total}),
                "normalized": normalized,
                "quality_score_obj": quality_score,
                "quality_score": float(quality_score.total),
                "quality_components": dict(quality_score.components),
                "visual_hash": deduplicator.compute_phash(normalized),
                "glare_x": glare_x,
                "glare_y": glare_y,
                "sharpness": float(quality_score.components.get("sharpness", 0.0)),
                "glare_mask": _compress_array(_glare_mask(normalized)),
                "laplacian_heatmap": _compress_array(_laplacian_heatmap(normalized)),
                "detection_id": int(c["detection_id"]),
                "frame_index": int(c["frame_index"]),
                "timestamp_ms": int(c.get("timestamp_ms", 0)),
                "confidence": float(c.get("confidence", 0.0)),
                "corners": [(float(x), float(y)) for x, y in (c.get("corners") or [])],
                "novelty_score": float(c.get("novelty_score", 1.0)),
                "width": int(c.get("width", 0)),
                "height": int(c.get("height", 0)),
                "triage_metrics": c.get("triage_metrics", {}),
                "is_canonical": False,
            })

        if not frame_entries:
            continue

        # Canonical selection — _select_canonical_entries expects a list of
        # dicts shaped like the V4 patched_entries (key: "candidate" is a
        # ScoredCandidate). We already set "candidate" above.
        canonical_entries = _select_canonical_entries(frame_entries, deduplicator)
        canonical_det_ids = {e["candidate"].detection_id for e in canonical_entries}
        for fe in frame_entries:
            if fe["detection_id"] in canonical_det_ids:
                fe["is_canonical"] = True
        best = max(canonical_entries, key=lambda e: e["quality_score"])

        best_canonical_img = best["normalized"]

        # ReID embedding (v5.5: array variant, no temp file)
        reid_embedding: Optional[List[float]] = track_dict.get("reid_embedding")
        if reid_embedding is not None and isinstance(reid_embedding, np.ndarray):
            reid_embedding = reid_embedding.tolist()
            
        if reid_embedding is None:
            embedder = _get_embedder()
            if embedder is not None:
                try:
                    emb_tensor = embedder.embed_array(best_canonical_img)
                    reid_embedding = emb_tensor.cpu().numpy().tolist()[0]
                except Exception as exc:
                    telemetry.resource_sample(
                        {"event": "reid_embedding_failed",
                         "instance_id": instance_id, "error": repr(exc)}
                    )

        # Track telemetry — per canonical entry. Repo is injected into
        # state by the runtime; tests can stub via state["repos"]["cards"].
        cards_repo = (state.get("repos") or {}).get("cards")
        if cards_repo is not None:
            for entry in canonical_entries:
                sc = entry["candidate"]
                if sc.corners:
                    try:
                        from card_capture.stages.score.selector import _get_polygon_area, _aspect_ratio
                        area = _get_polygon_area(sc.corners)
                        aspect = _aspect_ratio(sc.corners)
                        cx = sum(p[0] for p in sc.corners) / 4.0
                        cy = sum(p[1] for p in sc.corners) / 4.0
                        cards_repo.add_track_telemetry(
                            video_id=int(state.get("video_id", 0)),
                            instance_id=instance_id,
                            frame_index=int(sc.frame_index),
                            area=float(area),
                            aspect=float(aspect),
                            cx=float(cx),
                            cy=float(cy),
                        )
                    except Exception:
                        pass

        refined_tracks.append({
            "instance_id": instance_id,
            "track_id": int(track_dict.get("track_id", 0)),
            "angle": track_dict.get("angle", "Unknown"),
            "session_id": int(track_dict.get("session_id", 0)),
            "first_frame_index": int(track_dict.get("first_frame_index", -1)),
            "frame_entries": frame_entries,
            "canonical_detection_ids": list(canonical_det_ids),
            "best_canonical_detection_id": int(best["candidate"].detection_id),
            "best_canonical_image": best_canonical_img,
            "reid_embedding": reid_embedding,
        })
        
        pct = int(100 * (i + 1) / total_tracks)
        telemetry.progress("refine", pct, f"track {i+1}/{total_tracks}")

    state["refined_tracks"] = refined_tracks
    emit_stage_metrics(state, stage="refine", metrics={"refined_tracks": len(refined_tracks)})
