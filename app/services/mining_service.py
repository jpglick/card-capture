"""Service layer for dataset mining and active learning.

Provides methods for identifying high-value training candidates (hard cases)
and promoting them to the permanent training set.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from card_capture.data.connection import read_connection


class MiningService:
    def __init__(self, db_path: Path, training_data_dir: Path, training_repo=None):
        self.db_path = db_path
        self.training_data_dir = training_data_dir
        self._repo = training_repo

    def list_hard_cases(self, stage_id: Optional[str] = None) -> List[dict[str, Any]]:
        """Return all captured hard cases from the database."""
        query = "SELECT case_id, video_id, run_id, stage_id, reason, thumbnail_path, source_frame_path, created_at FROM hard_cases"
        params = []
        if stage_id:
            query += " WHERE stage_id = ?"
            params.append(stage_id)
        query += " ORDER BY created_at DESC"
        
        with read_connection(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
            keys = ("case_id", "video_id", "run_id", "stage_id", "reason",
                    "thumbnail_path", "source_frame_path", "created_at")
            return [dict(zip(keys, r)) for r in rows]

    def promote_to_training(self, case_id: int, model_name: str, label: str) -> Path:
        """Move a hard case image to the permanent training directory."""
        with read_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT thumbnail_path, source_frame_path FROM hard_cases WHERE case_id = ?",
                (case_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Case {case_id} not found")
                
            source_path = Path(row[0] or row[1])
            if not source_path.exists():
                raise FileNotFoundError(f"Source image missing: {source_path}")

            # Create destination: <training_data_dir>/<model_name>/<label>/<filename>
            dest_dir = self.training_data_dir / model_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            dest_path = dest_dir / f"case_{case_id}_{source_path.name}"
            shutil.copy(source_path, dest_path)
            
            # Update DB status
            # We need the writer here.
            if self._repo and self._repo._writer:
                from card_capture.data.writer import Write
                self._repo._writer.submit(Write(
                    "UPDATE hard_cases SET reason = ? WHERE case_id = ?",
                    (f"promoted:{model_name}:{label}", case_id)
                ))
            else:
                from card_capture.data.connection import open_connection
                conn_w = open_connection(self.db_path)
                try:
                    conn_w.execute(
                        "UPDATE hard_cases SET reason = ? WHERE case_id = ?",
                        (f"promoted:{model_name}:{label}", case_id)
                    )
                    conn_w.commit()
                finally:
                    conn_w.close()
            
            return dest_path

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Return counts of images in the permanent training directory."""
        stats = {}
        if not self.training_data_dir.exists():
            return stats
            
        for model_dir in self.training_data_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                stats[model_name] = {}
                for label_dir in model_dir.iterdir():
                    if label_dir.is_dir():
                        stats[model_name][label_dir.name] = len(list(label_dir.glob("*")))
        return stats
