"""Service layer for labeling operations.

Handles truth.json persistence in the database, front/back labels, and
dedup cluster verification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any

from card_capture.data.connection import read_connection
from harness.schema import TruthFile


def _to_file_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    try:
        return "/files/" + str(p.relative_to("card_capture_output"))
    except ValueError:
        pass
    parts = p.parts
    try:
        idx = parts.index("card_capture_output")
        rel = "/".join(parts[idx + 1:])
        return f"/files/{rel}" if rel else None
    except ValueError:
        return None


class LabelingService:
    def __init__(self, db_path: Path, labeling_repo=None) -> None:
        self.db_path = db_path
        self._repo = labeling_repo

    def get_truth(self, video_id: str) -> Optional[dict]:
        """Retrieve the truth.json payload for a video."""
        if self._repo:
            return self._repo.get_truth_payload(video_id)
        
        with read_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload_json FROM truth_files WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_truth(self, video_id: str, payload: dict) -> None:
        """Store or update the truth.json payload for a video."""
        # Validate against schema before saving
        tf = TruthFile.model_validate(payload)
        
        if self._repo:
            self._repo.store_truth_payload(video_id, tf.model_dump())
            return

        # Legacy fallback (to be removed once all callers use repo)
        from card_capture.data.writer import Write
        # We need a writer here... but service doesn't have one directly.
        # This is why we must use the repository.
        raise RuntimeError("LabelingService.put_truth requires labeling_repo")

    def post_fb_label(
        self,
        instance_id: str,
        frame_index: int,
        side: str,
        labeler: Optional[str] = None,
        source_run_id: Optional[int] = None,
    ) -> None:
        """Record a human (or model) front/back label."""
        if side not in ("front", "back", "uncertain", "no_card"):
            raise ValueError(f"Invalid side: {side}")
            
        if self._repo:
            self._repo.store_fb_label(
                instance_id=instance_id,
                frame_index=frame_index,
                side=side,
                labeler=labeler or "human",
                source_run_id=source_run_id,
            )
            return
        
        raise RuntimeError("LabelingService.post_fb_label requires labeling_repo")

    def next_fb_candidate(self) -> Optional[dict[str, Any]]:
        """Find the next high-confidence unlabeled detection for the F/B trainer.
        
        Returns:
            dict: The candidate data, or None if no unlabeled candidates remain.
        """
        with read_connection(self.db_path) as conn:
            card_cols = {r[1] for r in conn.execute("PRAGMA table_info(card_instances)").fetchall()}
            instance_expr = "ci.instance_id" if "instance_id" in card_cols else "ci.track_id"
            # Join card_views with fb_labels to find unlabeled instances.
            # We join card_views -> card_instances -> videos to get run_id/video_id.
            row = conn.execute(
                f"""
                SELECT
                    {instance_expr} AS instance_id,
                    cv.frame_index,
                    cv.rectified_path AS canonical_url,
                    v.source_path AS video_id,
                    ci.run_id
                FROM card_views cv
                JOIN card_instances ci ON ci.id = cv.card_instance_id
                JOIN videos v ON v.id = ci.video_id
                LEFT JOIN fb_labels fl ON fl.instance_id = {instance_expr}
                WHERE cv.is_canonical = 1 
                  AND fl.label_id IS NULL
                ORDER BY cv.confidence DESC
                LIMIT 1
                """
            ).fetchone()
        
        if not row:
            return None

        with read_connection(self.db_path) as conn:
            labels_collected = conn.execute("SELECT COUNT(*) FROM fb_labels").fetchone()[0]
            # Count distinct unlabeled instances that have a canonical view
            pending_count = conn.execute(f"""
                SELECT COUNT(DISTINCT {instance_expr})
                FROM card_views cv
                JOIN card_instances ci ON ci.id = cv.card_instance_id
                LEFT JOIN fb_labels fl ON fl.instance_id = {instance_expr}
                WHERE cv.is_canonical = 1 AND fl.label_id IS NULL
            """).fetchone()[0]

        res = {
            "instance_id": row[0],
            "frame_index": row[1],
            "canonical_url": _to_file_url(row[2]),
            "video_id": row[3],
            "run_id": row[4],
            "labels_collected": labels_collected,
            "labels_target": 500,
            "pending_count": pending_count,
        }
        return res

    def list_clusters(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        """List dedup clusters, optionally filtered by status."""
        query = "SELECT cluster_id, predicted_member_ids_json, confirmed_member_ids_json, status, updated_at FROM dedup_clusters"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC"
        
        with read_connection(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
            
        return [
            {
                "cluster_id": r[0],
                "predicted": json.loads(r[1]),
                "confirmed": json.loads(r[2]) if r[2] else None,
                "status": r[3],
                "updated_at": r[4],
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
        # This update should also go through a repository if we had one for clusters.
        # For now, we use read_connection and the writer? 
        # Wait, I don't have a cluster repository.
        # I'll use the raw writer if I can, or just keep it as is if it's the only one.
        # Actually, all writes MUST go through the Writer.
        
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
        
        sql = f"UPDATE dedup_clusters SET {', '.join(updates)} WHERE cluster_id = ?"
        
        # We need access to the writer.
        if self._repo and self._repo._writer:
            from card_capture.data.writer import Write
            self._repo._writer.submit(Write(sql=sql, params=tuple(params)))
        else:
            raise RuntimeError("LabelingService.patch_cluster requires repository with writer")
