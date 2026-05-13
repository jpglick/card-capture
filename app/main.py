"""Card Capture v4 — FastAPI application factory.

Usage (development):
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Usage (production-ish):
    python -m app.main
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI

from app.api import cards, config, events, label, regression, runs, training, videos
from app.services.event_bus import EventBus
from app.services.training_service import TrainingService


def create_app(db_path: Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db_path: Path to the SQLite database.  Defaults to ``cards.sqlite`` in
                 the current working directory.

    All configuration is injected here so tests can call ``create_app()``
    with a clean state every time.
    """
    if db_path is None:
        db_path = Path("cards.sqlite")

    if not db_path.exists():
        sqlite3.connect(db_path).close()

    app = FastAPI(
        title="Card Capture v4",
        version="0.1.0",
        description=(
            "REST + SSE service layer for the Card Capture trading-card "
            "extraction pipeline."
        ),
    )

    app.state.event_bus = EventBus()
    app.state.training_service = TrainingService(db_path=db_path)

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
