"""FastAPI application factory for Card Capture.

Usage::

    uvicorn app.main:app
    # or in tests:
    from app.main import create_app
    client = TestClient(create_app())
"""
import sqlite3
from pathlib import Path

from fastapi import FastAPI

from app.api.training import router as training_router
from app.services.training_service import TrainingService


def create_app(db_path: Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db_path: Path to the SQLite database.  Defaults to ``cards.sqlite`` in
                 the current working directory.  The file is created (empty) if
                 it does not exist; schema migrations are NOT run automatically
                 here — call :func:`migrations.run_migrations.apply_migrations`
                 separately if needed.
    """
    if db_path is None:
        db_path = Path("cards.sqlite")

    # Ensure the database file exists so the service can open it.
    if not db_path.exists():
        sqlite3.connect(db_path).close()

    application = FastAPI(title="Card Capture", version="0.1.0")
    application.state.training_service = TrainingService(db_path=db_path)
    application.include_router(training_router, prefix="/api/v1/training")
    return application


# Module-level app instance for ``uvicorn app.main:app``.
app = create_app()
