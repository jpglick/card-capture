from __future__ import annotations

from pathlib import Path

from card_capture.data.connection import read_connection
from card_capture.data.repositories.runs import RunsRepository
from card_capture.data.repositories.videos import VideosRepository
from card_capture.data.writer import Writer


def _video_id(prod_db: Path) -> int:
    writer = Writer(prod_db); writer.start()
    try:
        vid = VideosRepository(writer=writer, db_path=prod_db).register(
            source_path="/x.MOV", file_hash="h", duration_ms=1, width=100, height=100,
        )
        writer.flush()
        return vid
    finally:
        writer.stop()


def test_mark_started_then_completed(prod_db: Path) -> None:
    video_id = _video_id(prod_db)
    writer = Writer(prod_db); writer.start()
    try:
        repo = RunsRepository(writer=writer, db_path=prod_db)
        repo.mark_started(run_id="r1", video_id=video_id)
        repo.mark_completed(run_id="r1", cards_extracted=7)
        writer.flush()
        row = repo.get("r1")
    finally:
        writer.stop()

    assert row["run_id"] == "r1"
    assert row["video_id"] == video_id
    assert row["status"] == "completed"
    assert row["cards_extracted"] == 7
    assert row["finished_at"] is not None


def test_mark_failed_records_status(prod_db: Path) -> None:
    video_id = _video_id(prod_db)
    writer = Writer(prod_db); writer.start()
    try:
        repo = RunsRepository(writer=writer, db_path=prod_db)
        repo.mark_started("r2", video_id)
        repo.mark_failed("r2", error="boom")
        writer.flush()
        row = repo.get("r2")
    finally:
        writer.stop()

    assert row["status"] == "failed"
