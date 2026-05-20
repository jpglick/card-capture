"""RunPod serverless handler — deployed to RunPod, not run locally.

Deploy by building a Docker image with this file as the entrypoint:
    CMD ["python", "-m", "app.runpod_handler"]

R2 credentials are injected as RunPod endpoint environment variables:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
"""
from __future__ import annotations

import os
from pathlib import Path

import boto3
import runpod
from botocore.config import Config as BotocoreConfig

from app.worker_core import apply_cuda_config, restore_config, run_pipeline, package_results


def _r2_client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        region_name="auto",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=BotocoreConfig(s3={"addressing_style": "path"}),
    )


def handler(job: dict) -> dict:
    """RunPod calls this for each submitted job."""
    inp = job["input"]
    run_id = inp["run_id"]
    video_key = inp["video_r2_key"]
    results_key = inp["results_r2_key"]
    bucket = inp["r2_bucket"]
    config_preset = inp.get("config_preset", "balanced")

    s3 = _r2_client()

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
