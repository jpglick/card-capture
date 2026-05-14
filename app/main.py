"""Card Capture v4 — FastAPI application factory.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from app.api import cards, config, events, label, regression, runs, training, videos
from app.services.event_bus import EventBus
from app.services.training_service import TrainingService
from app.services.labeling_service import LabelingService
from app.services.regression_service import RegressionService
from app.services.video_service import VideoService
from app.services.runs_service import RunService
from app.services.cards_service import CardService
from app.services.playground_service import PlaygroundService
from app.services.mining_service import MiningService


def create_app(db_path: Optional[Path] = None) -> FastAPI:
    """Create and configure the FastAPI application.
    """
    if db_path is None:
        db_path = Path(os.environ.get("CC_DB", "card_capture_output/cards.sqlite"))

    # Ensure the database file exists.
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(db_path).close()

    # 1. Initialise storage tables (creates pipeline_events, card_instances, etc. if needed)
    from card_capture.storage import Storage
    storage = Storage(db_path)
    storage.initialize()

    # 2. Run migrations (applies v4 schema: truth_files, regression_baselines, etc.)
    from migrations.run_migrations import apply_migrations, assert_migrations_complete
    apply_migrations(db_path)
    assert_migrations_complete(db_path)

    app = FastAPI(
        title="Card Capture v4",
        version="0.1.0",
        description=(
            "REST + SSE service layer for the Card Capture trading-card "
            "extraction pipeline."
        ),
    )

    # Initialize services
    app.state.db_path = db_path
    app.state.event_bus = EventBus()
    app.state.training_service = TrainingService(db_path=db_path)
    app.state.labeling_service = LabelingService(db_path=db_path)
    app.state.regression_service = RegressionService(db_path=db_path)
    app.state.video_service = VideoService(db_path=db_path)
    app.state.run_service = RunService(db_path=db_path)
    app.state.card_service = CardService(db_path=db_path)
    app.state.playground_service = PlaygroundService(db_path=db_path)
    app.state.mining_service = MiningService(
        db_path=db_path,
        training_data_dir=Path("data/training")
    )

    # Include routers
    app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
    app.include_router(runs.router, prefix="/api/v1/runs", tags=["runs"])
    app.include_router(cards.router, prefix="/api/v1/cards", tags=["cards"])
    app.include_router(label.router, prefix="/api/v1/label", tags=["label"])
    app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
    app.include_router(regression.router, prefix="/api/v1/regression", tags=["regression"])
    app.include_router(config.router, prefix="/api/v1/config", tags=["config"])
    app.include_router(events.router, prefix="/events", tags=["events"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
