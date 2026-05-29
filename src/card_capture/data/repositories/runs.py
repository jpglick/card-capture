"""pipeline_runs repository."""
from __future__ import annotations

from pathlib import Path

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class RunsRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def mark_started(self, run_id: str, video_id: int) -> None:
        self._writer.submit(Write(
            sql="""
                INSERT OR REPLACE INTO pipeline_runs(run_id, video_id, status,
                                                     cards_extracted, started_at)
                VALUES (?, ?, 'running', 0, datetime('now'))
            """,
            params=(run_id, video_id),
        ))

    def mark_completed(self, run_id: str, cards_extracted: int) -> None:
        self._writer.submit(Write(
            sql="""
                UPDATE pipeline_runs
                SET status='completed',
                    cards_extracted=?,
                    finished_at=datetime('now')
                WHERE run_id=?
            """,
            params=(cards_extracted, run_id),
        ))

    def mark_failed(self, run_id: str, error: str | None = None) -> None:
        # `error` accepted for forward-compat; current schema discards it.
        self._writer.submit(Write(
            sql="""
                UPDATE pipeline_runs
                SET status='failed', finished_at=datetime('now')
                WHERE run_id=?
            """,
            params=(run_id,),
        ))

    def get(self, run_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT run_id, video_id, status, cards_extracted, started_at, finished_at "
                "FROM pipeline_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        keys = ("run_id", "video_id", "status", "cards_extracted",
                "started_at", "finished_at")
        return dict(zip(keys, row))
