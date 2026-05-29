"""training repository (production schema: migrations/0005_training_samples.sql)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class TrainingRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    # ------------------------------------------------------------------
    # Samples
    # ------------------------------------------------------------------

    def next_presence_sample(self) -> dict | None:
        with read_connection(self._db_path) as conn:
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
            "id": row[0],
            "image_path": row[1],
            "frame_index": row[2],
            "pending_count": pending,
        }

    def label_presence(self, sample_id: int, label: str) -> None:
        if self._writer is None:
            raise RuntimeError("TrainingRepository requires a Writer for updates")
        self._writer.submit(Write(
            sql="UPDATE presence_samples SET label=?, labeled_at=datetime('now') WHERE id=?",
            params=(label, sample_id),
        ))

    def next_corner_sample(self) -> dict | None:
        with read_connection(self._db_path) as conn:
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
            "id": row[0],
            "image_path": row[1],
            "frame_index": row[2],
            "predicted_corners": row[3],
            "confidence": row[4],
            "pending_count": pending,
        }

    def label_corner(self, sample_id: int, label: str, corrected_corners: str | None = None) -> None:
        if self._writer is None:
            raise RuntimeError("TrainingRepository requires a Writer for updates")
        self._writer.submit(Write(
            sql="UPDATE corner_samples SET label=?, corrected_corners=?, labeled_at=datetime('now') WHERE id=?",
            params=(label, corrected_corners, sample_id),
        ))

    # ------------------------------------------------------------------
    # Model Versions
    # ------------------------------------------------------------------

    def record_model_version(self, *, name: str, hash: str, metrics: Mapping[str, object], path: str) -> None:
        if self._writer is None:
            raise RuntimeError("TrainingRepository requires a Writer for updates")
        self._writer.submit(Write(
            sql="""
                INSERT INTO model_versions (model_name, training_set_hash, eval_metrics_json, checkpoint_path)
                VALUES (?, ?, ?, ?)
            """,
            params=(name, hash, json.dumps(dict(metrics)), path),
        ))

    def get_latest_model(self, name: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT model_name, training_set_hash, eval_metrics_json, checkpoint_path, created_at "
                "FROM model_versions WHERE model_name=? ORDER BY created_at DESC LIMIT 1",
                (name,),
            ).fetchone()
        if row is None:
            return None
        keys = ("model_name", "training_set_hash", "eval_metrics_json", "checkpoint_path", "created_at")
        return dict(zip(keys, row))

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def snapshot_baseline(self, job_id: str, n: int = 3) -> None:
        if self._writer is None:
            raise RuntimeError("TrainingRepository requires a Writer for updates")
        # This implementation requires a subquery or multi-step to get the run_ids.
        # Since we're in the repository, we can do it in two steps.
        with read_connection(self._db_path) as conn:
            runs = conn.execute(
                "SELECT run_id, cards_extracted FROM pipeline_runs "
                "WHERE status='completed' ORDER BY started_at DESC LIMIT ?",
                (n,),
            ).fetchall()
            
        for run_id, cards in runs:
            self._writer.submit(Write(
                sql="INSERT INTO benchmark_snapshots (job_id, run_id, cards_extracted) VALUES (?, ?, ?)",
                params=(job_id, run_id, cards),
            ))

    def get_benchmark_baseline(self, job_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT run_id, cards_extracted FROM benchmark_snapshots WHERE job_id=?",
                (job_id,),
            ).fetchall()
        return [{"run_id": r[0], "cards_extracted": r[1]} for r in rows]
