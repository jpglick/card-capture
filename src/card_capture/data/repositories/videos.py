"""videos repository — production schema (migrations/0001_v4_schema.sql)."""
from __future__ import annotations

from pathlib import Path

from card_capture.data.connection import open_connection, read_connection
from card_capture.data.writer import Writer, Write


class VideosRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def register(
        self,
        source_path: str,
        file_hash: str = "unknown",
        duration_ms: int = 0,
        width: int = 0,
        height: int = 0,
        status: str = "processing",
    ) -> int:
        """Insert a videos row and return the autoincrement id.

        Synchronous because callers immediately need the id for FK references.
        Uses a direct connection (not the writer queue) under the writer's
        thread lock to preserve single-writer semantics.
        """
        # Direct synchronous write through the writer's queue, blocking until
        # we have the row id. The writer worker is the only thread holding a
        # write connection; for inserts that must return autoincrement ids we
        # bypass the queue with the writer's lock held.
        with self._writer.serialize():  # see Task C.3 for Writer.serialize
            conn = open_connection(self._db_path)
            try:
                cur = conn.execute(
                    "INSERT INTO videos(source_path, file_hash, duration_ms, "
                    "width, height, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (source_path, file_hash, duration_ms, width, height, status),
                )
                conn.commit()
                return int(cur.lastrowid)
            finally:
                conn.close()

    def update_status(self, video_id: int, status: str) -> None:
        self._writer.submit(Write(
            sql="UPDATE videos SET status=? WHERE id=?",
            params=(status, video_id),
        ))

    def get(self, video_id: int) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT id, source_path, file_hash, duration_ms, width, height, "
                "status, created_at FROM videos WHERE id=?",
                (video_id,),
            ).fetchone()
        if row is None:
            return None
        keys = ("id", "source_path", "file_hash", "duration_ms", "width",
                "height", "status", "created_at")
        return dict(zip(keys, row))

    def list_recent(self, limit: int = 50) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, source_path, file_hash, duration_ms, width, height, "
                "status, created_at FROM videos ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("id", "source_path", "file_hash", "duration_ms", "width",
                "height", "status", "created_at")
        return [dict(zip(keys, r)) for r in rows]
