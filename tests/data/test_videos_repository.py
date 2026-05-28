from pathlib import Path

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.videos import VideosRepository


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE videos(
            video_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            registered_at_ms INTEGER NOT NULL,
            metadata TEXT
        )
    """)
    conn.close()


def test_register_and_get(tmp_path):
    db = tmp_path / "v.db"; _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = VideosRepository(writer=writer, db_path=db)
        repo.register("v1", "/path/to/v1.MOV", {"duration_s": 60})
        writer.flush()
        row = repo.get("v1")
    finally:
        writer.stop()
    assert row["video_id"] == "v1"
    assert row["metadata"]["duration_s"] == 60
