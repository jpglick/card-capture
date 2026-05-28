from pathlib import Path
import json

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.events import EventsRepository


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE pipeline_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            video_id TEXT,
            stage TEXT NOT NULL,
            elapsed_ms INTEGER NOT NULL,
            metadata TEXT
        )
    """)
    conn.close()


def test_record_stage_finished_and_list(tmp_path):
    db = tmp_path / "e.db"; _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = EventsRepository(writer=writer, db_path=db)
        repo.record_stage_finished("r1", "v1", "detect", 1234, {"frames": 100})
        repo.record_stage_finished("r1", "v1", "refine", 5678, {})
        writer.flush()
        rows = repo.list_for_run("r1")
    finally:
        writer.stop()
    assert [r["stage"] for r in rows] == ["detect", "refine"]
    assert json.loads(rows[0]["metadata"])["frames"] == 100
