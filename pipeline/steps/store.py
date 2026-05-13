"""Step 9 — store: persist final card records and views to the database.

Converts Metaflow artifacts into database records in ``card_instances``,
``card_views``, and ``saved_cards``.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from .start import RunContext

@dataclass
class StoreOutput:
    """Outputs of the store step."""
    final_cards: List[Dict[str, Any]]

def run(ctx: RunContext, groups: List[Dict[str, Any]], fused: List[Dict[str, Any]], prepared_tracks: List[Dict[str, Any]]) -> StoreOutput:
    """Persist results to the SQLite database.

    Args:
        ctx:             RunContext from the start step.
        groups:          Dedup groups from the dedup step.
        fused:           Fused image metadata from the fuse step.
        prepared_tracks: Full track data from the resolve step.

    Returns:
        ``StoreOutput`` containing the final saved card info.
    """
    from card_capture.storage import Storage
    from card_capture.models import CornerDetection
    import json
    
    storage = Storage(Path(ctx.db_path))
    
    video_id = ctx.video_id
    if video_id is None:
        raise ValueError("video_id missing from RunContext")

    # Map instance_id (UUID) to row_id (int)
    id_map: Dict[str, int] = {}
    
    # Map instance_id to its fused record
    fused_map: Dict[str, Dict[str, Any]] = {f["instance_id"]: f for f in fused}
    # Map instance_id to its prepared track
    track_map: Dict[str, Dict[str, Any]] = {t["instance_id"]: t for t in prepared_tracks}

    final_cards = []

    # 1. Store all card instances and their views
    for f in fused:
        iid = f["instance_id"]
        t = track_map[iid]
        
        # Convert reid_embedding list to bytes for BLOB
        embedding_bytes = None
        if f.get("reid_embedding") is not None:
            import numpy as np
            embedding_bytes = np.array(f["reid_embedding"], dtype=np.float32).tobytes()

        row_id = storage.add_card_instance(
            video_id=video_id,
            track_id=iid,
            angle=f["angle"],
            session_id=str(f["session_id"]),
            reid_embedding=embedding_bytes,
        )
        id_map[iid] = row_id
        
        # Update fused_image_path and primary_hash
        storage.update_instance_deduplication(
            row_id, f["primary_hash"], None, reid_embedding=embedding_bytes
        )
        storage.update_instance_fusion(row_id, f["fused_image_path"])

        # 2. Store card views for this instance
        best_det_id = t.get("best_canonical_detection_id")
        for fe in t.get("frame_entries", []):
            is_best = (fe["detection_id"] == best_det_id)
            
            # CRITICAL (A1): For the BEST canonical view, we point to the FUSED image
            view_path = f["fused_image_path"] if is_best else fe["image_path"]
            
            cd = CornerDetection(
                corners=fe["corners"],
                confidence=fe["confidence"],
            )
            
            view_id = storage.add_card_view(
                card_instance_id=row_id,
                frame_index=fe["frame_index"],
                timestamp_ms=fe["timestamp_ms"],
                detection=cd,
                rectified_path=view_path,
                quality_score=fe.get("quality_components"),
                is_canonical=fe.get("is_canonical", False),
                glare_x=fe.get("glare_x"),
                glare_y=fe.get("glare_y"),
                sharpness=fe.get("sharpness"),
                initial_confidence=fe.get("confidence"), # simplified
            )
            
            # Also add to legacy saved_cards if canonical for backward compatibility
            if fe.get("is_canonical") and is_best:
                storage.add_saved_card(
                    detection_id=view_id, 
                    image_path=view_path,
                    final_score=fe["quality_score"]
                )

    # 3. Store deduplication links
    for group in groups:
...
        canonical_iid = group["canonical_instance_id"]
        canonical_row_id = id_map[canonical_iid]
        
        for duplicate_iid in group["duplicate_instance_ids"]:
            duplicate_row_id = id_map[duplicate_iid]
            storage.update_instance_deduplication(
                duplicate_row_id, 
                fused_map[duplicate_iid]["primary_hash"], 
                canonical_row_id
            )

    # 3. Mark video as completed
    storage.update_video_status(video_id, "completed")

    return StoreOutput(final_cards=[])
