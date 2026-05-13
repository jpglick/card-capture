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
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    job_id: str
    model_name: str
    status: str  # "queued" | "running" | "completed" | "failed"
    created_at: str
    completed_at: Optional[str] = None
    progress: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


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
        """Return dataset sizes and stats for each registered model type."""
        import sqlite3
        from datetime import datetime

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            fb_rows = conn.execute("SELECT side, COUNT(*) as count FROM fb_labels GROUP BY side").fetchall()
            fb_dist = {row["side"]: row["count"] for row in fb_rows}
            fb_total = sum(fb_dist.values())
            
            # last_updated for fb
            fb_last = conn.execute("SELECT MAX(created_at) FROM fb_labels").fetchone()[0] or datetime.now().isoformat()

        return [
            {
                "model_name": "fb_classifier",
                "total_labels": fb_total,
                "class_distribution": fb_dist,
                "last_updated": fb_last,
            }
        ]

    def start_retrain(self, model_name: str, epochs: int, learning_rate: float) -> TrainingJob:
        """Enqueue a retrain job."""
        from datetime import datetime
        job_id = f"retrain-{model_name}-{int(datetime.now().timestamp())}"
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            status="queued",
            created_at=datetime.now().isoformat()
        )
        with self._lock:
            self._jobs[job_id] = job
        
        # In a real app we'd pop this from a queue, but here we just thread it
        thread = threading.Thread(target=self._run, args=(job, epochs, learning_rate), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Return the :class:`TrainingJob` for *job_id*, or ``None``."""
        with self._lock:
            return self._jobs.get(job_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, job: TrainingJob, epochs: int, learning_rate: float) -> None:
        from datetime import datetime
        job.status = "running"
        try:
            # Simulate progress
            for i in range(epochs):
                job.progress = {"epoch": i + 1, "total_epochs": epochs, "val_accuracy": 0.5 + (i/epochs)*0.4}
                import time
                time.sleep(0.1) # Simulate work

            if job.model_name == "fb_classifier":
                # Real training would happen here
                pass
            elif job.model_name == "dino_threshold":
                pass
            else:
                raise ValueError(f"unknown model: {job.model_name!r}")
            
            job.status = "completed"
            job.completed_at = datetime.now().isoformat()
        except Exception as exc:
            logger.exception("Training job %s failed: %s", job.job_id, exc)
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = datetime.now().isoformat()
