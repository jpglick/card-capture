"""Training service — in-process job queue for model retraining.

Provides :class:`TrainingService` which is attached to the FastAPI app's
``state`` object in :mod:`app.main`.  Routes in :mod:`app.api.training`
delegate to it via ``request.app.state.training_service``.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from card_capture.data.connection import read_connection

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    job_id: str
    model_name: str
    status: str  # "queued" | "running" | "completed" | "failed"
    created_at: str
    epochs: int = 30
    completed_at: Optional[str] = None
    progress: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    error: Optional[str] = None


class TrainingService:
    """Thread-safe, in-process job queue for ML model retraining."""

    def __init__(
        self,
        db_path: Path,
        training_repo=None,
        ml_repo=None,
    ) -> None:
        self.db_path = db_path
        self._training_repo = training_repo
        self._ml_repo = ml_repo
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[dict]:
        """Return dataset sizes and stats for each registered model type."""
        from datetime import datetime

        with read_connection(self.db_path) as conn:
            fb_rows = conn.execute("SELECT side, COUNT(*) as count FROM fb_labels GROUP BY side").fetchall()
            fb_dist = {row[0]: row[1] for row in fb_rows}
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

    def start_retrain(self, model_name: str, epochs: int = 30, learning_rate: float = 1e-3) -> TrainingJob:
        """Enqueue a retrain job.

        Raises:
            ValueError: If a job for this model is already queued or running.
        """
        from datetime import datetime

        with self._lock:
            for job in self._jobs.values():
                if job.model_name == model_name and job.status in ("queued", "running"):
                    raise ValueError(f"A training job for {model_name!r} is already in progress.")

            job_id = f"retrain-{model_name}-{int(datetime.now().timestamp())}"
            job = TrainingJob(
                job_id=job_id,
                model_name=model_name,
                status="queued",
                created_at=datetime.now().isoformat(),
                epochs=epochs,
            )
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Return the :class:`TrainingJob` for *job_id*, or ``None``."""
        with self._lock:
            return self._jobs.get(job_id)

    # ------------------------------------------------------------------
    # Presence queue
    # ------------------------------------------------------------------

    def next_presence_sample(self) -> Optional[dict]:
        if not self._training_repo:
            return None
        res = self._training_repo.next_presence_sample()
        if not res:
            return None
        return {
            "sample_id": res["id"],
            "image_url": self._to_url(res["image_path"]),
            "frame_index": res["frame_index"],
            "pending_count": res["pending_count"],
        }

    def label_presence(self, sample_id: int, label: str) -> None:
        assert label in ("present", "absent"), f"invalid label: {label!r}"
        if self._training_repo:
            self._training_repo.label_presence(sample_id, label)

    # ------------------------------------------------------------------
    # Corner queue
    # ------------------------------------------------------------------

    def next_corner_sample(self) -> Optional[dict]:
        if not self._training_repo:
            return None
        res = self._training_repo.next_corner_sample()
        if not res:
            return None
        return {
            "sample_id": res["id"],
            "image_url": self._to_url(res["image_path"]),
            "frame_index": res["frame_index"],
            "predicted_corners": res["predicted_corners"],
            "confidence": res["confidence"],
            "pending_count": res["pending_count"],
        }

    def label_corner(
        self,
        sample_id: int,
        label: str,
        corrected_corners: Optional[str] = None,
    ) -> None:
        assert label in ("correct", "adjusted", "negative"), f"invalid label: {label!r}"
        if self._training_repo:
            self._training_repo.label_corner(sample_id, label, corrected_corners)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        with read_connection(self.db_path) as conn:
            presence_pending = conn.execute(
                "SELECT COUNT(*) FROM presence_samples WHERE label IS NULL"
            ).fetchone()[0]
            fb_pending = conn.execute(
                """SELECT COUNT(*) FROM card_instances ci
                   WHERE NOT EXISTS (
                       SELECT 1 FROM fb_labels fl WHERE fl.instance_id = ci.track_id
                   )"""
            ).fetchone()[0]
            corner_pending = conn.execute(
                "SELECT COUNT(*) FROM corner_samples WHERE label IS NULL"
            ).fetchone()[0]

            accuracies = {}
            for model in ("presence", "fb_classifier"):
                row = conn.execute(
                    "SELECT eval_metrics_json FROM model_versions "
                    "WHERE model_name=? ORDER BY created_at DESC LIMIT 1",
                    (model,),
                ).fetchone()
                if row and row[0]:
                    import json
                    m = json.loads(row[0])
                    accuracies[model] = m.get("accuracy")

            history_rows = conn.execute(
                "SELECT model_name, eval_metrics_json, created_at FROM model_versions "
                "ORDER BY created_at ASC"
            ).fetchall()
            history = []
            for r in history_rows:
                if r[1]:
                    import json
                    m = json.loads(r[1])
                    history.append({
                        "model": r[0],
                        "accuracy": m.get("accuracy"),
                        "created_at": r[2],
                    })

        return {
            "pending": {
                "presence": presence_pending,
                "fb": fb_pending,
                "corners": corner_pending,
            },
            "accuracy": accuracies,
            "history": history,
        }

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def snapshot_baseline(self, job_id: str, n: int = 3) -> None:
        """Snapshot pipeline output for last N runs before retraining."""
        if self._training_repo:
            self._training_repo.snapshot_baseline(job_id, n)

    def get_benchmark_baseline(self, job_id: str) -> list[dict]:
        if self._training_repo:
            return self._training_repo.get_benchmark_baseline(job_id)
        return []

    def start_benchmark(self, n: int = 3) -> TrainingJob:
        from datetime import datetime
        job_id = f"benchmark-{int(datetime.now().timestamp())}"
        job = TrainingJob(
            job_id=job_id,
            model_name="benchmark",
            status="queued",
            created_at=datetime.now().isoformat(),
        )
        with self._lock:
            self._jobs[job_id] = job
        t = threading.Thread(target=self._run_benchmark_job, args=(job, n), daemon=True)
        t.start()
        return job

    def _run_benchmark_job(self, job: TrainingJob, n: int) -> None:
        from datetime import datetime
        try:
            with self._lock:
                job.status = "running"

            with read_connection(self.db_path) as conn:
                runs = conn.execute(
                    "SELECT run_id, video_id, cards_extracted FROM pipeline_runs "
                    "WHERE status='completed' ORDER BY started_at DESC LIMIT ?",
                    (n,),
                ).fetchall()
                videos = {
                    r[0]: r[1]
                    for r in conn.execute("SELECT id, source_path FROM videos").fetchall()
                }

            rows = []
            for run_id, video_id, cards_extracted in runs:
                video_path = videos.get(video_id, "")
                before = cards_extracted
                after = self._rerun_video(video_path)
                video_name = Path(video_path).name
                rows.append({
                    "video": video_name,
                    "before": before,
                    "after": after,
                    "delta": after - before,
                })

            with self._lock:
                job.status = "completed"
                job.completed_at = datetime.now().isoformat()
                job.progress = {"rows": rows}

        except Exception as exc:
            logger.exception("Benchmark job %s failed", job.job_id)
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = datetime.now().isoformat()

    def _rerun_video(self, video_path: str) -> int:
        import subprocess, sys, uuid
        from pathlib import Path as _Path
        run_id = f"benchmark-{uuid.uuid4().hex[:8]}"
        out_dir = _Path(self.db_path).parent / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "pipeline.card_capture_flow",
            "--no-pylint", "run",
            "--video", video_path,
            "--output-dir", str(out_dir),
            "--db", str(self.db_path),
            "--detector", "docaligner",
            "--config-preset", "balanced",
            "--ui-run-id", run_id,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"Pipeline failed: {proc.stderr[-500:]}")
        
        with read_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT cards_extracted FROM pipeline_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return row[0] if row else 0

    def get_benchmark_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status,
            "rows": job.progress.get("rows", []) if job.progress else [],
            "error": job.error,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_job(self, job: TrainingJob) -> None:
        from datetime import datetime
        try:
            with self._lock:
                job.status = "running"
                job.logs = [f"Starting {job.model_name} training ({job.epochs} epochs)…"]

            def _progress(p: dict):
                epoch = p.get("epoch", "?")
                total = p.get("total_epochs", "?")
                acc = p.get("val_accuracy", 0)
                line = f"Epoch {epoch}/{total}  val_acc={acc:.3f}"
                with self._lock:
                    job.progress = p
                    job.logs.append(line)

            if job.model_name == "presence":
                from card_capture.training.presence_trainer import train_presence
                metrics = train_presence(
                    db_path=self.db_path,
                    output_path=Path("models/presence_classifier.pt"),
                    epochs=job.epochs,
                    progress_cb=_progress,
                )
            elif job.model_name == "fb_classifier":
                from card_capture.training.fb_trainer import train_fb
                metrics = train_fb(
                    db_path=self.db_path,
                    output_path=Path("models/fb_classifier.pt"),
                    epochs=job.epochs,
                    progress_cb=_progress,
                )
            else:
                raise ValueError(f"unknown model: {job.model_name!r}")

            self._record_model_version(job.model_name, metrics)

            with self._lock:
                job.status = "completed"
                job.completed_at = datetime.now().isoformat()
                job.progress = metrics
                job.logs.append(f"Done — accuracy={metrics.get('accuracy', '?')}, val_samples={metrics.get('val_samples', '?')}")

        except Exception as exc:
            logger.exception("Training job %s failed", job.job_id)
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = datetime.now().isoformat()
                job.logs.append(f"FAILED: {exc}")

    def _record_model_version(self, model_name: str, metrics: dict) -> None:
        if self._training_repo:
            import time
            training_set_hash = str(int(time.time()))
            self._training_repo.record_model_version(
                name=model_name,
                hash=training_set_hash,
                metrics=metrics,
                path=f"models/{model_name}.pt",
            )

    def _to_url(self, abs_path: str) -> str:
        p = Path(abs_path)
        output_dir = Path(self.db_path).parent
        try:
            rel = p.relative_to(output_dir)
            return "/files/" + str(rel)
        except ValueError:
            return "/files/" + p.name
