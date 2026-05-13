"""Service layer for labeling operations.

Handles truth.json persistence in the database, front/back labels, and
dedup cluster verification.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, Any

from harness.schema import TruthFile


class LabelingService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def get_truth(self, video_id: str) -> Optional[dict]:
        """Retrieve the truth.json payload for a video."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT payload_json FROM truth_files WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def put_truth(self, video_id: str, payload: dict) -> None:
        """Store or update the truth.json payload for a video."""
        # Validate against schema before saving
        tf = TruthFile.model_validate(payload)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO truth_files(video_id, schema_version, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    payload_json=excluded.payload_json,
                    updated_at=datetime('now')
                """,
                (video_id, tf.schema_version, tf.model_dump_json()),
            )
            conn.commit()

    def post_fb_label(
        self,
        instance_id: str,
        frame_index: int,
        side: str,
        labeler: Optional[str] = None,
        source_run_id: Optional[int] = None,
    ) -> int:
        """Record a human (or model) front/back label."""
        if side not in ("front", "back", "uncertain"):
            raise ValueError(f"Invalid side: {side}")
            
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.execute(
                """
                INSERT INTO fb_labels(source_run_id, instance_id, frame_index, side, labeler)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_run_id, instance_id, frame_index, side, labeler),
            )
            conn.commit()
            return cur.lastrowid

    def next_fb_candidate(self) -> Optional[dict[str, Any]]:
        """Find the next high-confidence unlabeled detection for the F/B trainer."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            # Join card_views with fb_labels to find unlabeled instances
            row = conn.execute(
                """
                SELECT
                    cv.card_instance_id AS instance_id,
                    cv.frame_index,
                    cv.image_path,
                    cv.confidence
                FROM card_views cv
                LEFT JOIN fb_labels fl ON fl.instance_id = cv.card_instance_id
                WHERE fl.label_id IS NULL
                ORDER BY cv.confidence DESC
                LIMIT 1
                """
            ).fetchone()
        
        if not row:
            return None
            
        return {
            "instance_id": row["instance_id"],
            "frame_index": row["frame_index"],
            "image_path": row["image_path"],
            "confidence": float(row["confidence"]),
        }

    def list_clusters(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        """List dedup clusters, optionally filtered by status."""
        query = "SELECT * FROM dedup_clusters"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            
        return [
            {
                "cluster_id": r["cluster_id"],
                "predicted": json.loads(r["predicted_member_ids_json"]),
                "confirmed": json.loads(r["confirmed_member_ids_json"]) if r["confirmed_member_ids_json"] else None,
                "status": r["status"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def patch_cluster(
        self,
        cluster_id: int,
        *,
        status: Optional[str] = None,
        confirmed: Optional[list[str]] = None,
    ) -> None:
        """Update status or confirmed members of a dedup cluster."""
        updates = []
        params = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if confirmed is not None:
            updates.append("confirmed_member_ids_json = ?")
            params.append(json.dumps(confirmed))
        
        if not updates:
            return
            
        updates.append("updated_at = datetime('now')")
        params.append(cluster_id)
        
        query = f"UPDATE dedup_clusters SET {', '.join(updates)} WHERE cluster_id = ?"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(query, params)
            conn.commit()
