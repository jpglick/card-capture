"""Service layer for video management.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, List, Optional


class VideoService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def list_videos(self) -> List[dict[str, Any]]:
        """Return a list of all registered videos."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, source_path, status, created_at FROM videos ORDER BY created_at DESC"
            ).fetchall()
            return [
                {
                    "video_id": str(r["id"]),
                    "filename": Path(r["source_path"]).name,
                    "duration_ms": 0, # Placeholder
                    "status": r["status"],
                    "created_at": r["created_at"]
                }
                for r in rows
            ]

    def get_video(self, video_id: int) -> Optional[dict[str, Any]]:
        """Retrieve a single video record by ID."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, source_path, status, created_at FROM videos WHERE id = ?",
                (video_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "video_id": str(row["id"]),
                "filename": Path(row["source_path"]).name,
                "source_path": row["source_path"],
                "duration_ms": 0, # Placeholder
                "status": row["status"],
                "created_at": row["created_at"]
            }

    def add_video(self, source_path: str) -> int:
        """Register a new video for processing."""
        from card_capture.storage import Storage
        storage = Storage(self.db_path)
        
        # We don't have metadata yet, so we pass defaults
        # Real metadata should be probed before calling this if possible
        video_id = storage.add_video(
            source_path=source_path,
            file_hash="pending", # To be computed by pipeline
            duration_ms=0,
            width=0,
            height=0,
        )
        return video_id

    def delete_video(self, video_id: int) -> None:
        """Remove a video and its associated runs/cards."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Cascading deletes should be handled by FKs if enabled
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            conn.commit()
