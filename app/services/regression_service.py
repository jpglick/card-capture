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

    def compare(self, a_name: str, b_name: str) -> dict[str, Any]:
        """Compare two baselines and return deltas."""
        a = get_baseline(db_path=self.db_path, name=a_name)
        b = get_baseline(db_path=self.db_path, name=b_name)
        
        deltas = self._compute_deltas(a.metrics, b.metrics)
        
        return {
            "baseline_a": a_name,
            "baseline_b": b_name,
            "metrics_a": a.metrics,
            "metrics_b": b.metrics,
            "deltas": deltas,
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
