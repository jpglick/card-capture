from __future__ import annotations

from pathlib import Path

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.runs import RunsRepository


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE pipeline_runs(
            run_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            state TEXT NOT NULL,
            started_at_ms INTEGER,
            completed_at_ms INTEGER,
            cards_extracted INTEGER,
            error TEXT
        )
    """)
    conn.close()


def test_mark_started_then_completed(tmp_path):
    db = tmp_path / "r.db"
    _init_schema(db)
    writer = Writer(db)
    writer.start()
    try:
        repo = RunsRepository(writer=writer, db_path=db)
        repo.mark_started(run_id="r1", video_id="v1")
        repo.mark_completed(run_id="r1", cards_extracted=12)
        writer.flush()
        row = repo.get("r1")
    finally:
        writer.stop()
    assert row["state"] == "completed"
    assert row["cards_extracted"] == 12


def test_mark_failed_records_error(tmp_path):
    db = tmp_path / "rf.db"
    _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = RunsRepository(writer=writer, db_path=db)
        repo.mark_started("r2", "v2")
        repo.mark_failed("r2", error="boom")
        writer.flush()
        row = repo.get("r2")
    finally:
        writer.stop()
    assert row["state"] == "failed"
    assert row["error"] == "boom"
