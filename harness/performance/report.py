"""Comparison and aggregation helpers for perf reports. Filled out in later phases."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from harness.performance.runner import PerfReport  # noqa: F401  (re-export)


def load_reports(paths: Iterable[Path]) -> list[dict]:
    import json
    return [json.loads(p.read_text()) for p in paths]
