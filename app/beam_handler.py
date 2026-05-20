"""Beam endpoint handler — deployed to Beam, not run locally.

Deploy with:
    beam deploy app/beam_handler.py:process_video

The Volume ID and GPU type are configured at deploy time.
"""
from __future__ import annotations

from pathlib import Path

import beam

from app.worker_core import apply_cuda_config, restore_config, run_pipeline, package_results


@beam.endpoint(
    cpu=4,
    memory="16Gi",
    gpu="A10G",
)
def process_video(
    run_id: str,
    video_volume_path: str,
    results_volume_path: str,
    config_preset: str = "balanced",
) -> dict:
    """Run the card-capture pipeline on a video file stored in a Beam Volume."""
    output_dir = Path(f"/tmp/cc_output/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = apply_cuda_config()
    try:
        db_path = run_pipeline(run_id, video_volume_path, config_preset, output_dir)
        tarball_bytes = package_results(run_id, output_dir, db_path)
    finally:
        restore_config(original)

    results_path = Path(results_volume_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_bytes(tarball_bytes)
    return {"status": "complete", "results_path": results_volume_path}
