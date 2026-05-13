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
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """Return a paginated list of extracted card instances."""
        query = "SELECT * FROM card_instances"
        count_query = "SELECT COUNT(*) FROM card_instances"
        params = []
        where_clauses = []
        
        if run_id:
            where_clauses.append("(run_id = ? OR (run_id IS NULL AND 'legacy-' || video_id = ?))")
            params.extend([run_id, run_id])
        if video_id:
            where_clauses.append("video_id = ?")
            params.append(video_id)
            
        if where_clauses:
            clause = " WHERE " + " AND ".join(where_clauses)
            query += clause
            count_query += clause
            
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute(count_query, params).fetchone()[0]
            
            # Add limit and offset
            rows_params = list(params)
            rows_params.extend([page_size, (page - 1) * page_size])
            rows = conn.execute(query, rows_params).fetchall()
            
            items = []
            for r in rows:
                # Get the canonical view to get confidence
                view = conn.execute(
                    "SELECT confidence, rectified_path FROM card_views WHERE card_instance_id = ? AND is_canonical = 1",
                    (r["id"],)
                ).fetchone()
                
                vid = r["video_id"]
                video_row = conn.execute("SELECT source_path FROM videos WHERE id = ?", (vid,)).fetchone()
                video_id_str = Path(video_row["source_path"]).stem if video_row else str(vid)

                items.append({
                    "card_id": r["track_id"], # v4 uses track_id as stable card_id within run
                    "instance_id": str(r["id"]),
                    "video_id": video_id_str,
                    "run_id": r["run_id"] or f"legacy-{vid}",
                    "side": r["angle"] or "front",
                    "is_foil": False, # TODO: integrate with foil detection
                    "confidence": view["confidence"] if view else 0.0,
                    "review_state": "pending",
                    "canonical_url": view["rectified_path"] if view else None,
                    "fused_url": r["fused_image_path"],
                    "created_at": r["created_at"],
                })
            
            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "items": items,
            }

    def get_card(self, card_instance_id: int) -> Optional[dict[str, Any]]:
        """Retrieve a single card record by internal database ID."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM card_instances WHERE id = ?",
                (card_instance_id,),
            ).fetchone()
            if not row:
                return None
            
            view = conn.execute(
                "SELECT confidence, rectified_path, frame_index FROM card_views WHERE card_instance_id = ? AND is_canonical = 1",
                (row["id"],)
            ).fetchone()

            source_frames = conn.execute(
                "SELECT frame_index FROM card_views WHERE card_instance_id = ?",
                (row["id"],)
            ).fetchall()
            
            vid = row["video_id"]
            video_row = conn.execute("SELECT source_path FROM videos WHERE id = ?", (vid,)).fetchone()
            video_id_str = Path(video_row["source_path"]).stem if video_row else str(vid)

            return {
                "card_id": row["track_id"],
                "instance_id": str(row["id"]),
                "video_id": video_id_str,
                "run_id": row["run_id"] or f"legacy-{vid}",
                "side": row["angle"] or "front",
                "is_foil": False,
                "confidence": view["confidence"] if view else 0.0,
                "review_state": "pending",
                "canonical_url": view["rectified_path"] if view else None,
                "fused_url": row["fused_image_path"],
                "created_at": row["created_at"],
                "source_frame_indices": [f["frame_index"] for f in source_frames],
                "quality_score": {
                    "total": view["confidence"] if view else 0.0, # Placeholder
                }
            }
