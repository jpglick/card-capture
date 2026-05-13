"""Service layer for card management.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, List, Optional


class CardService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def list_cards(
        self,
        run_id: Optional[str] = None,
        video_id: Optional[int] = None,
    ) -> List[dict[str, Any]]:
        """Return a list of extracted card instances."""
        query = "SELECT * FROM card_instances"
        params = []
        where_clauses = []
        
        if run_id:
            # card_instances doesn't have run_id, but it has video_id
            # We'd need to join or map run_id to video_id
            pass
        if video_id:
            where_clauses.append("video_id = ?")
            params.append(video_id)
            
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
            
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "instance_id": r["track_id"], # UUID
                    "run_id": "unknown", # To be filled
                    "angle": r["angle"] or "Front",
                    "fused_image_url": f"/static/crops/{Path(r['fused_image_path']).name}" if r["fused_image_path"] else "",
                    "quality_score": 0.0, # To be filled
                    "primary_hash": r["visual_hash"] or "",
                }
                for r in rows
            ]

    def get_card(self, instance_id: int) -> Optional[dict[str, Any]]:
        """Retrieve a single card record by ID."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM card_instances WHERE id = ?",
                (instance_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "instance_id": row["track_id"],
                "run_id": "unknown",
                "angle": row["angle"] or "Front",
                "fused_image_url": f"/static/crops/{Path(row['fused_image_path']).name}" if row["fused_image_path"] else "",
                "quality_score": 0.0,
                "primary_hash": row["visual_hash"] or "",
            }
