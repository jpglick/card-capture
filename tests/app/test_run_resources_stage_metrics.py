from __future__ import annotations

from pathlib import Path

from app.services.runs_service import RunService
from card_capture.data.repositories.events import EventsRepository
from card_capture.data.repositories.runs import RunsRepository
from card_capture.data.repositories.videos import VideosRepository
from card_capture.data.writer import Writer
from migrations.run_migrations import apply_migrations


def test_get_run_resources_includes_stage_metrics(tmp_path: Path) -> None:
    db = tmp_path / "cards.sqlite"
    apply_migrations(db)

    writer = Writer(db)
    writer.start()
    try:
        videos = VideosRepository(writer=writer, db_path=db)
        runs = RunsRepository(writer=writer, db_path=db)
        events = EventsRepository(writer=writer, db_path=db)

        video_id = videos.register(
            source_path="/tmp/x.MOV",
            file_hash="h",
            duration_ms=1000,
            width=10,
            height=10,
        )
        runs.mark_started("r1", video_id=video_id)
        events.record_stage_metrics(
            run_id="r1",
            video_id=video_id,
            stage="detect",
            metrics={"detections": 21},
        )
        writer.flush()
    finally:
        writer.stop()

    resources = RunService(db).get_run_resources("r1")
    assert resources["stage_metrics"]["detect"] == {"detections": 21}

