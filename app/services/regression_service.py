"""Service layer for regression reporting.

Provides methods for listing baselines and comparing runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.baseline import get_baseline, list_baselines


class RegressionService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def list_baselines(self) -> list[dict[str, Any]]:
        """Return a list of all regression baselines."""
        return list_baselines(db_path=self.db_path)

    def compare(self, run_a: int, run_b: int) -> dict[str, Any]:
        """Compare two regression runs and return deltas."""
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            a = conn.execute("SELECT * FROM regression_runs WHERE run_id = ?", (run_a,)).fetchone()
            b = conn.execute("SELECT * FROM regression_runs WHERE run_id = ?", (run_b,)).fetchone()
            
            if not a or not b:
                raise ValueError("One or both runs not found")
                
            metrics_a = json.loads(a["metrics_json"])
            metrics_b = json.loads(b["metrics_json"])
            
            deltas = self._compute_deltas(metrics_a, metrics_b)
            
            # Identify regressions (where delta is negative and significant)
            regressions = []
            for k, v in deltas.items():
                if isinstance(v, (int, float)) and v < -0.01: # 1% threshold
                    regressions.append(f"{k} dropped by {abs(v):.2%}")
            
            # Per-video deltas
            pv_a = {v["video_id"]: v for v in json.loads(a["per_video_json"])}
            pv_b = {v["video_id"]: v for v in json.loads(b["per_video_json"])}
            
            per_video_deltas = []
            for vid in set(pv_a.keys()) | set(pv_b.keys()):
                ma = pv_a.get(vid, {}).get("metrics", {})
                mb = pv_b.get(vid, {}).get("metrics", {})
                vdeltas = self._compute_deltas(ma, mb)
                
                per_video_deltas.append({
                    "video_id": vid,
                    "deltas": vdeltas,
                    "is_regression": any(v < -0.01 for v in vdeltas.values() if isinstance(v, (int, float)))
                })

            return {
                "run_a": run_a,
                "run_b": run_b,
                "metric_deltas": deltas,
                "regressions": regressions,
                "per_video_deltas": per_video_deltas,
            }

    def _compute_deltas(self, metrics_a: dict, metrics_b: dict) -> dict[str, Any]:
        """Compute deltas between two metric dictionaries."""
        deltas = {}
        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
        
        for k in all_keys:
            va = metrics_a.get(k)
            vb = metrics_b.get(k)
            
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                deltas[k] = vb - va
            elif isinstance(va, dict) and isinstance(vb, dict):
                # Recurse for nested metrics (e.g. DedupAccuracy as dict)
                deltas[k] = self._compute_deltas(va, vb)
            else:
                deltas[k] = None
                
        return deltas
