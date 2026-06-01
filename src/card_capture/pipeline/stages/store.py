"""Stage 10b: Storage via repositories.

Port of V4 ``pipeline/steps/store.py``. All ``Storage`` calls replaced
with ``CardsRepository`` methods (Phase 2). Image writes happen here
and ONLY here, matching the V5.5 in-memory mandate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from card_capture.pipeline.stage_metrics import emit_stage_metrics


def run(state: dict, *, telemetry) -> None:
    request = state["request"]
    run_id = request.run_id
    video_id = int(state.get("video_id", 0))
    if video_id == 0:
        telemetry.contract_violation(
            "store_without_video_id", {"hint": "runtime must inject video_id"}
        )
        raise RuntimeError("store stage reached without a real video_id")

    cards_repo = state["repos"]["cards"]
    runs_repo = state["repos"]["runs"]
    fused_canonicals: List[Dict[str, Any]] = state.get("fused_canonicals", [])
    prepared_tracks: List[Dict[str, Any]] = state.get("prepared_tracks", [])
    dedup_groups: List[Dict[str, Any]] = state.get("dedup_groups", [])

    output_root: Path = state["output_root"]
    crops_dir = output_root / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Write all images to disk. After this block every fused_canonical
    #    has a ``fused_image_path`` and every frame_entry has an
    #    ``image_path`` — the DB writes below reference these paths.
    # ------------------------------------------------------------------
    for fused in fused_canonicals:
        iid = fused["instance_id"]
        path = crops_dir / f"instance_{iid[:8]}_fused.jpg"
        cv2.imwrite(str(path), fused["fused_image"])
        fused["fused_image_path"] = str(path)

    fused_by_iid: Dict[str, Dict[str, Any]] = {f["instance_id"]: f for f in fused_canonicals}
    track_by_iid: Dict[str, Dict[str, Any]] = {t["instance_id"]: t for t in prepared_tracks}

    for track in prepared_tracks:
        iid = track["instance_id"]
        for fe in track.get("frame_entries", []):
            det_id = int(fe["detection_id"])
            view_path = crops_dir / f"track_{iid[:8]}_det_{det_id}_rectified.jpg"
            cv2.imwrite(str(view_path), fe["normalized"])
            fe["image_path"] = str(view_path)

    # ------------------------------------------------------------------
    # 2. Persist card_instances + card_views via the repository.
    # ------------------------------------------------------------------
    id_map: Dict[str, int] = {}
    final_cards: List[Dict[str, Any]] = []

    for fused in fused_canonicals:
        iid = fused["instance_id"]
        track = track_by_iid.get(iid, {})

        embedding_bytes: bytes | None = None
        if fused.get("reid_embedding") is not None:
            embedding_bytes = np.array(fused["reid_embedding"], dtype=np.float32).tobytes()
        else:
            # V4 fallback: compute from the fused image so the row never
            # has a NULL embedding (used by cross-video dedup in future runs).
            try:
                from card_capture.ml.embeddings import compute_reid_embedding_array
                emb = compute_reid_embedding_array(fused["fused_image"])
                embedding_bytes = emb.astype(np.float32).tobytes()
            except Exception as exc:
                cards_repo.add_pipeline_event(
                    video_id=video_id, frame_index=0, timestamp_ms=0,
                    event_type="reid_embedding_failed",
                    data={"instance_id": iid, "error": repr(exc)},
                )

        row_id = cards_repo.add_card_instance(
            video_id=video_id,
            track_id=iid,
            angle=fused["angle"],
            session_id=str(fused["session_id"]),
            reid_embedding=embedding_bytes,
            run_id=run_id,
        )
        id_map[iid] = row_id

        cards_repo.update_instance_deduplication(
            row_id=row_id,
            primary_hash=fused["primary_hash"],
            cross_video_parent=None,
            reid_embedding=embedding_bytes,
        )
        cards_repo.update_instance_fusion(
            row_id=row_id,
            fused_image_path=fused["fused_image_path"],
        )

        best_det_id = int(track.get("best_canonical_detection_id", -1))
        for fe in track.get("frame_entries", []):
            is_best = int(fe["detection_id"]) == best_det_id
            view_path = fused["fused_image_path"] if is_best else fe["image_path"]

            view_id = cards_repo.add_card_view(
                card_instance_id=row_id,
                frame_index=int(fe["frame_index"]),
                timestamp_ms=int(fe["timestamp_ms"]),
                corners=fe["corners"],
                confidence=float(fe["confidence"]),
                rectified_path=view_path,
                quality_score=fe.get("quality_components", {}),
                is_canonical=bool(fe.get("is_canonical", False)),
                glare_x=fe.get("glare_x"),
                glare_y=fe.get("glare_y"),
                sharpness=fe.get("sharpness"),
                initial_confidence=float(fe["confidence"]),
            )
            if fe.get("is_canonical") and is_best:
                cards_repo.add_saved_card(
                    detection_id=view_id,
                    image_path=view_path,
                    final_score=float(fe["quality_score"]),
                    video_id=video_id,
                    source_path=view_path,
                    timestamp_ms=int(fe["timestamp_ms"]),
                    score_components_json=fe.get("quality_components", {}),
                )

        final_cards.append({
            "instance_id": iid,
            "row_id": row_id,
            "video_id": video_id,
            "run_id": run_id,
            "angle": fused["angle"],
            "fused_image_path": fused["fused_image_path"],
        })

    # ------------------------------------------------------------------
    # 3. Persist dedup links (intra-run + cross-video).
    # ------------------------------------------------------------------
    for group in dedup_groups:
        canonical_iid = group["canonical_instance_id"]
        if canonical_iid not in id_map:
            continue
        canonical_row_id = id_map[canonical_iid]

        cross_parent = group.get("cross_video_parent_id")
        if cross_parent:
            cards_repo.update_instance_deduplication(
                row_id=canonical_row_id,
                primary_hash=fused_by_iid[canonical_iid]["primary_hash"],
                cross_video_parent=int(cross_parent),
            )
        for dup_iid in group["duplicate_instance_ids"]:
            if dup_iid not in id_map:
                continue
            cards_repo.update_instance_deduplication(
                row_id=id_map[dup_iid],
                primary_hash=fused_by_iid[dup_iid]["primary_hash"],
                cross_video_parent=canonical_row_id,
            )

    state["cards"] = final_cards
    state["output_artifacts"] = [str(crops_dir / "*.jpg")]
    runs_repo.mark_completed(run_id, cards_extracted=len(final_cards))
    emit_stage_metrics(state, stage="store", metrics={"final_cards": len(final_cards)})
