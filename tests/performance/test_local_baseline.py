"""Real-video baseline. Run manually:
    python3 -m pytest tests/performance/test_local_baseline.py -m benchmark
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.performance.runner import run


BASELINE_VIDEO = Path(os.environ.get("V55_BASELINE_VIDEO", "/nonexistent.MOV"))


@pytest.mark.benchmark
@pytest.mark.skipif(not BASELINE_VIDEO.exists(), reason="V55_BASELINE_VIDEO not set or missing")
def test_local_v55_runs_in_one_process(tmp_path):
    report = run(profile="local_v55", video=str(BASELINE_VIDEO), out_dir=tmp_path)
    assert report.error is None
    assert report.counters["video_reopens"] == 0, "refine must not re-open the video"
    assert report.counters["model_loads"] <= 4, "models must load once per run"
