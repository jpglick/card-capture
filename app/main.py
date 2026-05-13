"""FastAPI application factory for Card Capture.

Usage::

    uvicorn app.main:app
    # or in tests:
    from app.main import create_app
    client = TestClient(create_app())
"""
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from app.api.training import router as training_router
from app.api.label import router as label_router
from app.api.regression import router as regression_router
from app.services.training_service import TrainingService
from app.services.labeling_service import LabelingService
from app.services.regression_service import RegressionService


def create_app(db_path: Optional[Path] = None) -> FastAPI:
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

    # Ensure the database file exists.
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(db_path).close()

    # 1. Initialise storage tables (creates pipeline_events, card_instances, etc. if needed)
    from card_capture.storage import Storage
    storage = Storage(db_path)
    storage.initialize()

    # 2. Run migrations (applies v4 schema: truth_files, regression_baselines, etc.)
    from migrations.run_migrations import apply_migrations
    apply_migrations(db_path)

    application = FastAPI(title="Card Capture", version="0.1.0")
    
    # Initialize services
    application.state.training_service = TrainingService(db_path=db_path)
    application.state.labeling_service = LabelingService(db_path=db_path)
    application.state.regression_service = RegressionService(db_path=db_path)
    
    # Include routers
    application.include_router(training_router, prefix="/api/v1/training")
    application.include_router(label_router, prefix="/api/v1/label")
    application.include_router(regression_router, prefix="/api/v1/regression")
    
    return application


# Module-level app instance for ``uvicorn app.main:app``.
app = create_app()
