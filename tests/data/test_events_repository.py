from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.events import EventsRepository
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


def test_record_stage_finished_persists(prod_db: Path) -> None:
    video_id = _video_id(prod_db)
    writer = Writer(prod_db); writer.start()
    try:
        repo = EventsRepository(writer=writer, db_path=prod_db)
        repo.record_stage_finished(
            run_id="r1",
            video_id=video_id,
            stage="detect",
            frame_index=42,
            timestamp_ms=1_700_000_000_000,
            elapsed_ms=12,
            metadata={"corners_found": 4},
        )
        writer.flush()
        events = repo.list_for_run("r1")
    finally:
        writer.stop()

    assert len(events) == 1
    e = events[0]
    assert e["stage_id"] == "detect"
    assert e["frame_index"] == 42
    assert e["event_type"] == "stage_finished"
    assert e["video_id"] == video_id
