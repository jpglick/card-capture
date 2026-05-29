"""Smoke test: the perf harness produces a well-formed JSON report."""
from __future__ import annotations

import json
from pathlib import Path

from harness.performance.runner import run


def test_synthetic_smoke_writes_report(tmp_path: Path):
    report = run(profile="synthetic_smoke", video="none", out_dir=tmp_path)
    assert report.success is True if not hasattr(report, "error") else report.error is None
    
    report_file = tmp_path / "perf_report.json"
    assert report_file.exists()
    
    data = json.loads(report_file.read_text())
    assert data["profile"] == "synthetic_smoke"
    assert "timings_ms" in data
    assert "__total__" in data["timings_ms"]
