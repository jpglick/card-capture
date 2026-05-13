"""Service layer for dataset mining and active learning.

Provides methods for identifying high-value training candidates (hard cases)
and promoting them to the permanent training set.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class MiningService:
    def __init__(self, db_path: Path, training_data_dir: Path):
        self.db_path = db_path
        self.training_data_dir = training_data_dir

    def list_hard_cases(self, stage_id: Optional[str] = None) -> List[dict[str, Any]]:
        """Return all captured hard cases from the database."""
        query = "SELECT * FROM hard_cases"
        params = []
        if stage_id:
            query += " WHERE stage_id = ?"
            params.append(stage_id)
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def promote_to_training(self, case_id: int, model_name: str, label: str) -> Path:
        """Move a hard case image to the permanent training directory."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM hard_cases WHERE case_id = ?", (case_id,)).fetchone()
            if not row:
                raise ValueError(f"Case {case_id} not found")
                
            source_path = Path(row["thumbnail_path"] or row["source_frame_path"])
            if not source_path.exists():
                raise FileNotFoundError(f"Source image missing: {source_path}")

            # Create destination: <training_data_dir>/<model_name>/<label>/<filename>
            dest_dir = self.training_data_dir / model_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            dest_path = dest_dir / f"case_{case_id}_{source_path.name}"
            shutil.copy(source_path, dest_path)
            
            # Update DB status
            conn.execute(
                "UPDATE hard_cases SET reason = ? WHERE case_id = ?",
                (f"promoted:{model_name}:{label}", case_id)
            )
            conn.commit()
            
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
