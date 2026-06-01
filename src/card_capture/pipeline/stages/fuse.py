"""Stage 9: Lighting-Diverse Fusion.

V5.5 change: was a Metaflow ``foreach`` (one subprocess per track, ~4–6
minutes overhead on the reference video). Now runs the fusion loop
in-process with a plain ``for``. Algorithm unchanged — see
:mod:`card_capture.stages.fuse.fuser`.

Substitutions vs V4 ``pipeline/steps/fuse.py``:
- ``cv2.imread(fe['image_path'])`` -> ``fe['normalized']``
- ``cv2.imwrite(fused_path, fused_img)`` -> ``fused['fused_image'] = img``
- ``shutil.copy(best_path, fused_path)`` (single-frame path) -> direct
  passthrough of ``prepared_track['best_canonical_image']``
"""
from __future__ import annotations

from typing import Any, Dict, List

from card_capture.shared.stage_metrics import emit_stage_metrics


def run(state: dict, *, telemetry) -> None:
    config = state["request"].config
    prepared_tracks = state.get("prepared_tracks") or []

    fusion_target = int(config.get("fusion_target_frames", 1))
    enable_foil = bool(config.get("enable_foil_aware_fusion", True))
    foil_threshold = float(config.get("foil_threshold", 50.0))

    fused_canonicals: List[Dict[str, Any]] = []

    from card_capture.stages.fuse.fuser import MultiFrameFuser

    total_tracks = len(prepared_tracks)
    for i, track in enumerate(prepared_tracks):
        instance_id = track["instance_id"]
        frame_entries = track.get("frame_entries", [])
        canonical_entries = [fe for fe in frame_entries if fe.get("is_canonical")]

        if not canonical_entries:
            # Fallback: use best_canonical_image directly
            fused_img = track.get("best_canonical_image")
            primary_hash = (frame_entries[0].get("visual_hash", "") if frame_entries else "")
            quality_score = float(frame_entries[0].get("quality_score", 0.0) if frame_entries else 0.0)
        else:
            images = [fe["normalized"] for fe in canonical_entries if fe.get("normalized") is not None]
            primary_hash = str(canonical_entries[0].get("visual_hash", ""))
            quality_score = float(canonical_entries[0].get("quality_score", 0.0))

            if not images:
                fused_img = track.get("best_canonical_image")
            elif len(images) == 1 or fusion_target <= 1:
                # Single-frame passthrough
                fused_img = track.get("best_canonical_image", images[0])
            else:
                try:
                    fuser = MultiFrameFuser()
                    fused_img = fuser.fuse(
                        images,
                        foil_threshold=foil_threshold if enable_foil else None,
                    )
                except Exception as exc:
                    telemetry.resource_sample(
                        {"event": "fusion_failed",
                         "instance_id": instance_id, "error": repr(exc)}
                    )
                    fused_img = track.get("best_canonical_image", images[0])

        fused_canonicals.append({
            "instance_id": instance_id,
            "session_id": int(track.get("session_id", 0)),
            "angle": track.get("angle", "Unknown"),
            "fused_image": fused_img,
            "primary_hash": primary_hash,
            "quality_score": quality_score,
            "side_score": float(track.get("side_score", 0.0)),
            "appearance_vector": list(track.get("appearance_vector", [])),
            "best_canonical_detection_id": int(track.get("best_canonical_detection_id", 0)),
            "duplicate_track_index": track.get("duplicate_track_index"),
            "first_frame_index": int(track.get("first_frame_index", -1)),
            "reid_embedding": track.get("reid_embedding"),
        })

        pct = int(100 * (i + 1) / total_tracks)
        telemetry.progress("fuse", pct, f"track {i+1}/{total_tracks}")

    state["fused_canonicals"] = fused_canonicals
    emit_stage_metrics(state, stage="fuse", metrics={"fused_canonicals": len(fused_canonicals)})
