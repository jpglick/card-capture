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
    epochs: int = 30
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
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, image_path, frame_index FROM presence_samples "
                "WHERE label IS NULL ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            pending = conn.execute(
                "SELECT COUNT(*) FROM presence_samples WHERE label IS NULL"
            ).fetchone()[0]
        return {
            "sample_id": row["id"],
            "image_url": self._to_url(row["image_path"]),
            "frame_index": row["frame_index"],
            "pending_count": pending,
        }

    def label_presence(self, sample_id: int, label: str) -> None:
        assert label in ("present", "absent"), f"invalid label: {label!r}"
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE presence_samples SET label=?, labeled_at=datetime('now') WHERE id=?",
                (label, sample_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Corner queue
    # ------------------------------------------------------------------

    def next_corner_sample(self) -> Optional[dict]:
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, image_path, frame_index, predicted_corners, confidence "
                "FROM corner_samples WHERE label IS NULL ORDER BY confidence LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            pending = conn.execute(
                "SELECT COUNT(*) FROM corner_samples WHERE label IS NULL"
            ).fetchone()[0]
        return {
            "sample_id": row["id"],
            "image_url": self._to_url(row["image_path"]),
            "frame_index": row["frame_index"],
            "predicted_corners": row["predicted_corners"],
            "confidence": row["confidence"],
            "pending_count": pending,
        }

    def label_corner(
        self,
        sample_id: int,
        label: str,
        corrected_corners: Optional[str] = None,
    ) -> None:
        assert label in ("correct", "adjusted", "negative"), f"invalid label: {label!r}"
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """UPDATE corner_samples
                   SET label=?, corrected_corners=?, labeled_at=datetime('now')
                   WHERE id=?""",
                (label, corrected_corners, sample_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

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
                if row and row["eval_metrics_json"]:
                    import json
                    m = json.loads(row["eval_metrics_json"])
                    accuracies[model] = m.get("accuracy")

            history_rows = conn.execute(
                "SELECT model_name, eval_metrics_json, created_at FROM model_versions "
                "ORDER BY created_at ASC"
            ).fetchall()
            history = []
            for r in history_rows:
                if r["eval_metrics_json"]:
                    import json
                    m = json.loads(r["eval_metrics_json"])
                    history.append({
                        "model": r["model_name"],
                        "accuracy": m.get("accuracy"),
                        "created_at": r["created_at"],
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
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            runs = conn.execute(
                "SELECT run_id, cards_extracted FROM pipeline_runs "
                "ORDER BY started_at DESC LIMIT ?",
                (n,),
            ).fetchall()
            for run_id, cards in runs:
                conn.execute(
                    "INSERT INTO benchmark_snapshots (job_id, run_id, cards_extracted) "
                    "VALUES (?, ?, ?)",
                    (job_id, run_id, cards),
                )
            conn.commit()

    def get_benchmark_baseline(self, job_id: str) -> list[dict]:
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT run_id, cards_extracted FROM benchmark_snapshots WHERE job_id=?",
                (job_id,),
            ).fetchall()
        return [dict(r) for r in rows]

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
        import sqlite3
        try:
            with self._lock:
                job.status = "running"

            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                runs = conn.execute(
                    "SELECT run_id, video_id, cards_extracted FROM pipeline_runs "
                    "WHERE status='completed' ORDER BY started_at DESC LIMIT ?",
                    (n,),
                ).fetchall()
                videos = {
                    r["id"]: r["source_path"]
                    for r in conn.execute("SELECT id, source_path FROM videos").fetchall()
                }

            rows = []
            for run in runs:
                video_path = videos.get(run["video_id"], "")
                before = run["cards_extracted"]
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
        import subprocess, sys, uuid, sqlite3
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
        with sqlite3.connect(str(self.db_path)) as conn:
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

            def _progress(p: dict):
                with self._lock:
                    job.progress = p

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

        except Exception as exc:
            logger.exception("Training job %s failed", job.job_id)
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = datetime.now().isoformat()

    def _record_model_version(self, model_name: str, metrics: dict) -> None:
        import sqlite3, json
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO model_versions (model_name, training_set_hash, eval_metrics_json, checkpoint_path) "
                "VALUES (?, ?, ?, ?)",
                (
                    model_name,
                    "",
                    json.dumps(metrics),
                    f"models/{model_name}.pt",
                ),
            )
            conn.commit()

    def _to_url(self, abs_path: str) -> str:
        p = Path(abs_path)
        output_dir = Path(self.db_path).parent
        try:
            rel = p.relative_to(output_dir)
            return "/files/" + str(rel)
        except ValueError:
            return "/files/" + p.name
