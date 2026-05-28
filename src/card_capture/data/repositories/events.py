"""pipeline_events repository."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class EventsRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def record_stage_finished(
        self, run_id: str, video_id: str | None, stage: str,
        elapsed_ms: int, metadata: Mapping[str, object],
    ) -> None:
        self._writer.submit(Write(
            sql="""
                INSERT INTO pipeline_events(run_id, video_id, stage, elapsed_ms, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
            params=(run_id, video_id, stage, elapsed_ms, json.dumps(dict(metadata))),
        ))

    def list_for_run(self, run_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT run_id, video_id, stage, elapsed_ms, metadata FROM pipeline_events "
                "WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        keys = ("run_id", "video_id", "stage", "elapsed_ms", "metadata")
        return [dict(zip(keys, r)) for r in rows]
