"""Training service — in-process job queue for model retraining.

Provides :class:`TrainingService` which is attached to the FastAPI app's
``state`` object in :mod:`app.main`.  Routes in :mod:`app.api.training`
delegate to it via ``request.app.state.training_service``.
"""
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    job_id: str
    model_name: str
    status: str  # "queued" | "running" | "completed" | "failed"
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class TrainingService:
    """Thread-safe, in-process job queue for ML model retraining."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[dict]:
        """Return dataset sizes for each registered model type.

        Queries the ``fb_labels`` and ``dedup_clusters`` tables introduced by
        Contract 1 (migrations/0001_v4_schema.sql).
        """
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            fb = conn.execute("SELECT COUNT(*) FROM fb_labels").fetchone()[0]
            dedup = conn.execute(
                "SELECT COUNT(*) FROM dedup_clusters WHERE status='confirmed'"
            ).fetchone()[0]
        return [
            {"name": "fb_labels", "size": fb},
            {"name": "dedup_clusters", "size": dedup},
        ]

    def start_retrain(self, model_name: str, *, dry_run: bool = False) -> TrainingJob:
        """Enqueue a retrain job and return the :class:`TrainingJob` immediately.

        The actual training runs in a daemon thread so the HTTP request returns
        quickly.  Poll :meth:`get_job` to check status.
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job = TrainingJob(job_id=job_id, model_name=model_name, status="queued")
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job, dry_run), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> TrainingJob | None:
        """Return the :class:`TrainingJob` for *job_id*, or ``None``."""
        with self._lock:
            return self._jobs.get(job_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, job: TrainingJob, dry_run: bool) -> None:
        job.status = "running"
        try:
            if dry_run:
                job.metrics = {"dry_run": True}
            elif job.model_name == "fb_classifier":
                from card_capture.ml.training.fb_train import train as fb_train  # type: ignore[import]

                job.metrics = fb_train(db_path=self.db_path)
            elif job.model_name == "dino_threshold":
                from card_capture.ml.training.dedup_calibrate import calibrate  # type: ignore[import]

                job.metrics = calibrate(db_path=self.db_path)
            else:
                raise ValueError(f"unknown model: {job.model_name!r}")
            job.status = "completed"
        except Exception as exc:
            logger.exception("Training job %s failed: %s", job.job_id, exc)
            job.status = "failed"
            job.error = str(exc)
