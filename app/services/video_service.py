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
                "SELECT id, source_path, status, duration_ms, created_at FROM videos ORDER BY created_at DESC"
            ).fetchall()
            return [
                {
                    "video_id": str(r["id"]),
                    "filename": Path(r["source_path"]).name,
                    "duration_ms": r["duration_ms"],
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
                "SELECT id, source_path, status, duration_ms, created_at FROM videos WHERE id = ?",
                (video_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "video_id": str(row["id"]),
                "filename": Path(row["source_path"]).name,
                "source_path": row["source_path"],
                "duration_ms": row["duration_ms"],
                "status": row["status"],
                "created_at": row["created_at"]
            }

    def add_video(self, source_path: str) -> int:
        """Register a new video for processing."""
        from card_capture.storage import Storage
        from card_capture.ingestion import probe_video

        path = Path(source_path)
        if not path.exists():
            # If relative, try to find it in data/videos or golden_set/videos
            candidates = [
                Path("data/videos") / source_path,
                Path("golden_set/videos") / source_path,
            ]
            for cand in candidates:
                if cand.exists():
                    path = cand
                    break

        metadata = probe_video(path) if path.exists() else {}

        storage = Storage(self.db_path)
        video_id = storage.add_video(
            source_path=str(path),
            file_hash="pending",
            duration_ms=metadata.get("duration_ms", 0),
            width=metadata.get("width", 0),
            height=metadata.get("height", 0),
            status="pending",
        )
        return video_id

    def update_status(self, video_id: int, status: str) -> None:
        from card_capture.storage import Storage
        Storage(self.db_path).update_video_status(video_id, status)

    def delete_video(self, video_id: int) -> None:
        """Remove a video and its associated runs/cards."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Cascading deletes should be handled by FKs if enabled
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            conn.commit()
