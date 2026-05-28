"""videos repository."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class VideosRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def register(self, video_id: str, path: str, metadata: Mapping[str, object]) -> None:
        self._writer.submit(Write(
            sql="""
                INSERT OR REPLACE INTO videos(video_id, path, registered_at_ms, metadata)
                VALUES (?, ?, ?, ?)
            """,
            params=(video_id, path, int(time.time() * 1000), json.dumps(dict(metadata))),
        ))

    def get(self, video_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT video_id, path, registered_at_ms, metadata FROM videos WHERE video_id=?",
                (video_id,),
            ).fetchone()
            if row is None:
                return None
            return {"video_id": row[0], "path": row[1],
                    "registered_at_ms": row[2], "metadata": json.loads(row[3] or "{}")}

    def list_recent(self, limit: int = 50) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT video_id, path, registered_at_ms, metadata FROM videos "
                "ORDER BY registered_at_ms DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"video_id": r[0], "path": r[1], "registered_at_ms": r[2],
                 "metadata": json.loads(r[3] or "{}")} for r in rows]
