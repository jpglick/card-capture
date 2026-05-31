from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.events import EventsRepository
from card_capture.data.repositories.videos import VideosRepository
from card_capture.data.writer import Writer


def _video_id(prod_db: Path) -> int:
    writer = Writer(prod_db)
    writer.start()
    try:
        video_id = VideosRepository(writer=writer, db_path=prod_db).register(
            source_path="/x.MOV",
            file_hash="h",
            duration_ms=1,
            width=10,
            height=10,
        )
        writer.flush()
        return video_id
    finally:
        writer.stop()


def test_record_stage_metrics_persists(prod_db: Path) -> None:
    video_id = _video_id(prod_db)
    writer = Writer(prod_db)
    writer.start()
    try:
        repo = EventsRepository(writer=writer, db_path=prod_db)
        repo.record_stage_metrics(
            run_id="r1",
            video_id=video_id,
            stage="track",
            metrics={"tracks_final": 18, "tracks_data": 18},
        )
        writer.flush()
        rows = [e for e in repo.list_for_run("r1") if e["event_type"] == "stage_metrics"]
    finally:
        writer.stop()

    assert len(rows) == 1
    assert rows[0]["stage_id"] == "track"
    assert rows[0]["data"] == {"tracks_final": 18, "tracks_data": 18}

