"""Service layer for managing video metadata and processing status."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from card_capture.data.connection import read_connection


class VideoService:
    def __init__(self, db_path: Path, videos_repo=None) -> None:
        self.db_path = db_path
        self._repo = videos_repo

    def list_videos(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return a list of the most recent videos."""
        if self._repo:
            return [self._to_api_video(v) for v in self._repo.list_recent(limit=limit)]
            
        with read_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM videos ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        # Fallback for when repo is not yet fully utilized or returns different shape
        db_rows = [
            dict(
                zip(
                    ["id", "source_path", "file_hash", "duration_ms", "width", "height", "status", "created_at"],
                    r,
                )
            )
            for r in rows
        ]
        return [self._to_api_video(v) for v in db_rows]

    def get_video(self, video_id: int) -> Optional[dict[str, Any]]:
        """Retrieve metadata for a specific video."""
        if self._repo:
            row = self._repo.get(video_id)
            return self._to_api_video(row) if row else None
            
        with read_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE id = ?",
                (video_id,),
            ).fetchone()
        if row is None:
            return None
        db_row = dict(
            zip(
                ["id", "source_path", "file_hash", "duration_ms", "width", "height", "status", "created_at"],
                row,
            )
        )
        return self._to_api_video(db_row)

    def _to_api_video(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "video_id": str(row["id"]),
            "filename": Path(str(row["source_path"])).name,
            "duration_ms": int(row.get("duration_ms", 0) or 0),
            "status": str(row.get("status", "pending")),
            "created_at": str(row.get("created_at", "")),
            # Preserve internal fields used by callers like start_run.
            "source_path": str(row.get("source_path", "")),
            "id": int(row["id"]),
        }

    def add_video(
        self,
        source_path: str,
        file_hash: str = "unknown",
        duration_ms: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> int:
        """Register a new video for processing."""
        if self._repo:
            return self._repo.register(
                source_path=source_path,
                file_hash=file_hash,
                duration_ms=duration_ms,
                width=width,
                height=height,
                status="pending",
            )
            
        # Legacy fallback
        from card_capture.data.connection import open_connection
        conn = open_connection(self.db_path)
        try:
            cur = conn.execute(
                """
                INSERT INTO videos(source_path, file_hash, duration_ms, width, height)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_path, file_hash, duration_ms, width, height),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def update_status(self, video_id: int, status: str) -> None:
        """Update the processing status of a video."""
        if self._repo:
            self._repo.update_status(video_id, status)
            return

        from card_capture.data.connection import open_connection
        conn = open_connection(self.db_path)
        try:
            conn.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))
            conn.commit()
        finally:
            conn.close()

    def delete_video(self, video_id: int) -> None:
        """Remove a video and its associated data from the database."""
        # This should probably be in the repository, but for now we do it here.
        # It needs multiple writes, so it must go through the writer.
        if self._repo and self._repo._writer:
            from card_capture.data.writer import Write
            self._repo._writer.submit(Write("DELETE FROM videos WHERE id = ?", (video_id,)))
            self._repo._writer.submit(Write("DELETE FROM pipeline_events WHERE video_id = ?", (video_id,)))
            self._repo._writer.submit(Write("DELETE FROM card_instances WHERE video_id = ?", (video_id,)))
        else:
            from card_capture.data.connection import open_connection
            conn = open_connection(self.db_path)
            try:
                conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
                conn.commit()
            finally:
                conn.close()
