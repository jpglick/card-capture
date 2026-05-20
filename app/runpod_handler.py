"""RunPod serverless handler — deployed to RunPod, not run locally.

Deploy by building a Docker image with this file as the entrypoint:
    CMD ["python", "-m", "app.runpod_handler"]

RunPod injects AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as env vars
for the built-in S3 storage.
"""
from __future__ import annotations

from pathlib import Path

import boto3
import runpod

from app.worker_core import apply_cuda_config, restore_config, run_pipeline, package_results

# Confirm from https://docs.runpod.io/storage/s3-api before deploying
_RUNPOD_S3_ENDPOINT = "https://storage.runpod.io"


def handler(job: dict) -> dict:
    """RunPod calls this for each submitted job."""
    inp = job["input"]
    run_id = inp["run_id"]
    video_key = inp["video_s3_key"]
    results_key = inp["results_s3_key"]
    bucket = inp["bucket"]
    config_preset = inp.get("config_preset", "balanced")

    # Credentials are injected by RunPod as standard AWS env vars
    s3 = boto3.client("s3", endpoint_url=_RUNPOD_S3_ENDPOINT)

    video_path = Path(f"/tmp/{run_id}_input.mov")
    s3.download_file(bucket, video_key, str(video_path))

    output_dir = Path(f"/tmp/cc_output/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = apply_cuda_config()
    try:
        db_path = run_pipeline(run_id, str(video_path), config_preset, output_dir)
        tarball_bytes = package_results(run_id, output_dir, db_path)
    finally:
        restore_config(original)

    s3.put_object(Bucket=bucket, Key=results_key, Body=tarball_bytes)
    return {"status": "complete", "results_key": results_key}


runpod.serverless.start({"handler": handler})
