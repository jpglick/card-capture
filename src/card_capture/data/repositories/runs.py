"""pipeline_runs repository."""
from __future__ import annotations

import time
from pathlib import Path

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class RunsRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def mark_started(self, run_id: str, video_id: str) -> None:
        now = int(time.time() * 1000)
        self._writer.submit(Write(
            sql="""
                INSERT OR REPLACE INTO pipeline_runs(run_id, video_id, state, started_at_ms)
                VALUES (?, ?, 'started', ?)
            """,
            params=(run_id, video_id, now),
        ))

    def mark_completed(self, run_id: str, cards_extracted: int) -> None:
        now = int(time.time() * 1000)
        self._writer.submit(Write(
            sql="""
                UPDATE pipeline_runs
                SET state='completed', completed_at_ms=?, cards_extracted=?
                WHERE run_id=?
            """,
            params=(now, cards_extracted, run_id),
        ))

    def mark_failed(self, run_id: str, error: str) -> None:
        now = int(time.time() * 1000)
        self._writer.submit(Write(
            sql="""
                UPDATE pipeline_runs SET state='failed', completed_at_ms=?, error=? WHERE run_id=?
            """,
            params=(now, error, run_id),
        ))

    def get(self, run_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT run_id, video_id, state, started_at_ms, completed_at_ms, cards_extracted, error "
                "FROM pipeline_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            keys = ("run_id", "video_id", "state", "started_at_ms", "completed_at_ms", "cards_extracted", "error")
            return dict(zip(keys, row))
