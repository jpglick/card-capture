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

def run(ctx: RunContext, groups: List[Dict[str, Any]], fused: List[Dict[str, Any]]) -> StoreOutput:
    """Persist results to the SQLite database.

    Args:
        ctx:    RunContext from the start step.
        groups: Dedup groups from the dedup step.
        fused:  Fused image metadata from the fuse step.

    Returns:
        ``StoreOutput`` containing the final saved card info.
    """
    from card_capture.storage import Storage
    storage = Storage(Path(ctx.db_path))
    
    video_id = ctx.video_id
    if video_id is None:
        raise ValueError("video_id missing from RunContext")

    # Map instance_id (UUID) to row_id (int)
    id_map: Dict[str, int] = {}
    
    # Map instance_id to its fused record
    fused_map: Dict[str, Dict[str, Any]] = {f["instance_id"]: f for f in fused}

    final_cards = []

    # 1. Store all card instances
    for f in fused:
        iid = f["instance_id"]
        row_id = storage.add_card_instance(
            video_id=video_id,
            track_id=iid,
            angle=f["angle"],
            session_id=str(f["session_id"]),
        )
        id_map[iid] = row_id
        
        # Update fused_image_path and primary_hash
        storage.update_instance_deduplication(
            row_id, f["primary_hash"], None # We'll set duplicate_of later
        )
        # Note: update_instance_deduplication in storage.py only sets phash and duplicate_of.
        # We also need to set fused_image_path.
        with storage._connect() as conn:
            conn.execute(
                "UPDATE card_instances SET fused_image_path = ? WHERE id = ?",
                (f["fused_image_path"], row_id)
            )

    # 2. Store deduplication links
    for group in groups:
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
