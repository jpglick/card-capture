"""Card Capture v4 — FastAPI application factory.
"""
from __future__ import annotations

import os

# Load .env from the project root before anything else reads os.environ.
# Safe to call even if the file doesn't exist (no-op).
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import batch, cards, cdp, config, events, label, regression, runs, training, videos
from app.services.event_bus import EventBus
from app.services.training_service import TrainingService
from app.services.labeling_service import LabelingService
from app.services.regression_service import RegressionService
from app.services.video_service import VideoService
from app.services.runs_service import RunService
from app.services.cards_service import CardService
from app.services.playground_service import PlaygroundService
from app.services.mining_service import MiningService
from app.services.cdp_service import CdpService


_REPO_ROOT = Path(__file__).resolve().parent.parent


def create_app(db_path: Optional[Path] = None, output_root: Optional[Path] = None) -> FastAPI:
    """Build and configure the FastAPI application.
    """
    if db_path is None:
        db_path = Path(os.environ.get("CC_DB", "var/db/cards.sqlite"))

    # Ensure the database file exists.
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        from card_capture.data.connection import open_connection
        open_connection(db_path).close()

    # 1. Initialise storage tables (creates pipeline_events, card_instances, etc. if needed)
    from card_capture.stages.store.storage import Storage
    storage = Storage(db_path)
    storage.initialize()

    # 2. Run migrations (applies v4 schema: truth_files, regression_baselines, etc.)
    from migrations.run_migrations import apply_migrations, assert_migrations_complete
    apply_migrations(db_path)
    assert_migrations_complete(db_path)

    from contextlib import asynccontextmanager
    from card_capture.data.writer import Writer
    from card_capture.data.repositories.runs import RunsRepository
    from card_capture.data.repositories.events import EventsRepository
    from card_capture.data.repositories.cards import CardsRepository
    from card_capture.data.repositories.videos import VideosRepository
    from card_capture.data.repositories.labeling import LabelingRepository
    from card_capture.data.repositories.telemetry import TelemetryRepository
    from card_capture.data.repositories.config import ConfigRepository
    from card_capture.data.repositories.batch import BatchRepository
    from card_capture.data.repositories.training import TrainingRepository
    from card_capture.data.repositories.ml import MLRepository

    # 3. Instantiate repositories and services (writer=None initially)
    writer = Writer(db_path=db_path)
    runs_repo = RunsRepository(writer, db_path)
    events_repo = EventsRepository(writer, db_path)
    cards_repo = CardsRepository(writer, db_path)
    videos_repo = VideosRepository(writer, db_path)
    labeling_repo = LabelingRepository(writer, db_path)
    telemetry_repo = TelemetryRepository(writer, db_path)
    config_repo = ConfigRepository(writer, db_path)
    batch_repo = BatchRepository(writer, db_path)
    training_repo = TrainingRepository(writer, db_path)
    ml_repo = MLRepository(writer, db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start the single-writer thread
        app.state.writer.start()
        # Configure OpenTelemetry exporters once per process. Opt-in via env
        # (OTEL_EXPORTER_OTLP_ENDPOINT / CARD_CAPTURE_OTEL_CONSOLE); a no-op
        # otherwise. Without this the OTel sink records into a no-op provider.
        from card_capture.pipeline.otel_setup import configure_telemetry
        app.state.otel_enabled = configure_telemetry()
        if app.state.otel_enabled:
            import logging
            logging.getLogger("card_capture.app").info("OpenTelemetry export enabled")
        # A run still 'running' on a fresh process is orphaned — its worker was
        # killed (e.g. OOM during refine) before it could record a terminal
        # status. Reconcile to 'failed' so the UI stops polling it forever.
        reaped = app.state.runs_repo.reap_orphaned_runs()
        if reaped:
            import logging
            logging.getLogger("card_capture.app").warning(
                "Reaped %d orphaned run(s) left 'running' by a prior process: %s",
                len(reaped), ", ".join(reaped),
            )
        try:
            yield
        finally:
            app.state.writer.stop()

    app = FastAPI(
        title="Card Capture v4",
        version="0.1.0",
        description=(
            "REST + SSE service layer for the Card Capture trading-card "
            "extraction pipeline."
        ),
        lifespan=lifespan,
    )

    # Initialize state
    app.state.db_path = db_path
    app.state.writer = writer
    app.state.runs_repo = runs_repo
    app.state.events_repo = events_repo
    app.state.cards_repo = cards_repo
    app.state.videos_repo = videos_repo
    app.state.labeling_repo = labeling_repo
    app.state.telemetry_repo = telemetry_repo
    app.state.otel_enabled = False  # set by lifespan via configure_telemetry()
    app.state.config_repo = config_repo
    app.state.batch_repo = batch_repo
    app.state.training_repo = training_repo
    app.state.ml_repo = ml_repo

    app.state.event_bus = EventBus()
    app.state.training_service = TrainingService(
        db_path=db_path,
        training_repo=training_repo,
        ml_repo=ml_repo,
    )
    app.state.labeling_service = LabelingService(
        db_path=db_path,
        labeling_repo=labeling_repo,
    )
    app.state.regression_service = RegressionService(db_path=db_path)
    app.state.video_service = VideoService(
        db_path=db_path,
        videos_repo=videos_repo,
    )
    app.state.run_service = RunService(
        db_path=db_path,
        runs_repo=runs_repo,
        events_repo=events_repo,
    )
    app.state.card_service = CardService(
        db_path=db_path,
        cards_repo=cards_repo,
    )
    app.state.cdp_service = CdpService(db_path=db_path)
    app.state.playground_service = PlaygroundService(db_path=db_path)
    app.state.mining_service = MiningService(
        db_path=db_path,
        training_data_dir=Path("data/training"),
        training_repo=training_repo,
    )

    # Serve pipeline output (crops, frames) as static files. The pipeline writes
    # to <repo>/var/output/<run_id>/... (see app/api/videos.py) and card paths are
    # mapped to /files URLs rooted at var/output (see cards_service._to_file_url),
    # so /files MUST serve var/output — NOT db_path.parent (which is var/db).
    if output_root is None:
        output_root = Path(os.environ.get("CC_OUTPUT") or (_REPO_ROOT / "var/output"))
    output_root.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=str(output_root)), name="files")

    # Include routers
    app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
    app.include_router(batch.router, prefix="/api/v1/runs/batch", tags=["batch"])
    app.include_router(runs.router, prefix="/api/v1/runs", tags=["runs"])
    app.include_router(cards.router, prefix="/api/v1/cards", tags=["cards"])
    app.include_router(label.router, prefix="/api/v1/label", tags=["label"])
    app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
    app.include_router(regression.router, prefix="/api/v1/regression", tags=["regression"])
    app.include_router(config.router, prefix="/api/v1/config", tags=["config"])
    app.include_router(cdp.router, prefix="/api/v1/cdp", tags=["cdp"])
    app.include_router(events.router, prefix="/events", tags=["events"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
