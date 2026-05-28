from pathlib import Path
import json

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.telemetry import TelemetryRepository


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE telemetry_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            at_ms INTEGER NOT NULL
        )
    """)
    conn.close()


def test_record_and_list(tmp_path):
    db = tmp_path / "t.db"; _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = TelemetryRepository(writer=writer, db_path=db)
        repo.record_event(run_id="r1", kind="resource_sample", payload={"vram_mb": 4096})
        writer.flush()
        events = repo.list_for_run("r1")
    finally:
        writer.stop()
    assert events[0]["kind"] == "resource_sample"
    assert json.loads(events[0]["payload"])["vram_mb"] == 4096
