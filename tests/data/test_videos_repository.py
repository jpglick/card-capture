from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.videos import VideosRepository
from card_capture.data.writer import Writer


def test_register_and_get(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = VideosRepository(writer=writer, db_path=prod_db)
        video_id = repo.register(
            source_path="/abs/path/IMG_5872.MOV",
            file_hash="sha256:deadbeef",
            duration_ms=12_345,
            width=3840,
            height=2160,
        )
        writer.flush()
    finally:
        writer.stop()

    assert isinstance(video_id, int) and video_id > 0
    row = repo.get(video_id)
    assert row["source_path"] == "/abs/path/IMG_5872.MOV"
    assert row["file_hash"] == "sha256:deadbeef"
    assert row["duration_ms"] == 12_345
    assert row["width"] == 3840
    assert row["height"] == 2160
    assert row["status"] == "processing"


def test_list_recent_returns_newest_first(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = VideosRepository(writer=writer, db_path=prod_db)
        ids = [
            repo.register("/a.MOV", "h1", 1, 100, 100),
            repo.register("/b.MOV", "h2", 2, 100, 100),
            repo.register("/c.MOV", "h3", 3, 100, 100),
        ]
        writer.flush()
        recent = repo.list_recent(limit=2)
    finally:
        writer.stop()

    assert [r["id"] for r in recent] == [ids[-1], ids[-2]]
