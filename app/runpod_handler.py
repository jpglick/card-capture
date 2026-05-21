"""RunPod serverless handler — deployed to RunPod, not run locally.

Deploy by building a Docker image with this file as the entrypoint:
    CMD ["python", "-m", "app.runpod_handler"]

R2 credentials are injected as RunPod endpoint environment variables:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
"""
from __future__ import annotations

import os
import time
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


def _t(label: str, start: float) -> float:
    elapsed = time.time() - start
    print(f"[diag] {label}: {elapsed:.1f}s", flush=True)
    return time.time()


def handler(job: dict) -> dict:
    """RunPod calls this for each submitted job."""
    t0 = job_start = time.time()
    inp = job["input"]
    run_id = inp["run_id"]
    video_key = inp["video_r2_key"]
    results_key = inp["results_r2_key"]
    bucket = inp["r2_bucket"]
    config_preset = inp.get("config_preset", "balanced")

    print(f"[diag] handler started  run_id={run_id}  preset={config_preset}", flush=True)

    s3 = _r2_client()

    # ── Video download ───────────────────────────────────────────────────────
    video_path = Path(f"/tmp/{run_id}_input.mov")
    s3.download_file(bucket, video_key, str(video_path))
    video_mb = video_path.stat().st_size / 1_048_576
    t0 = _t(f"R2 download ({video_mb:.1f} MB)", t0)

    # ── Pipeline ─────────────────────────────────────────────────────────────
    output_dir = Path(f"/tmp/cc_output/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = apply_cuda_config()
    try:
        db_path = run_pipeline(run_id, str(video_path), config_preset, output_dir)
    finally:
        restore_config(original)
    t0 = _t("pipeline total", t0)

    # ── Diagnostics from DB ──────────────────────────────────────────────────
    _print_db_diagnostics(run_id, db_path, output_dir)

    # ── Package ──────────────────────────────────────────────────────────────
    tarball_bytes = package_results(run_id, output_dir, db_path)
    tarball_mb = len(tarball_bytes) / 1_048_576
    t0 = _t(f"package results ({tarball_mb:.1f} MB tarball)", t0)

    # ── Upload ───────────────────────────────────────────────────────────────
    s3.put_object(Bucket=bucket, Key=results_key, Body=tarball_bytes)
    t0 = _t(f"R2 upload", t0)

    total = time.time() - job_start
    print(f"[diag] TOTAL handler time: {total:.1f}s", flush=True)

    return {"status": "complete", "results_key": results_key}


def _print_db_diagnostics(run_id: str, db_path: Path, output_dir: Path) -> None:
    """Print card/frame counts from the pipeline DB."""
    import sqlite3 as _sqlite3
    if not db_path.exists():
        print("[diag] DB not found — pipeline produced no output", flush=True)
        return
    try:
        with _sqlite3.connect(db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

            def count(table: str, where: str = "") -> int:
                if table not in tables:
                    return -1
                q = f"SELECT COUNT(*) FROM {table}"
                if where:
                    q += f" WHERE {where}"
                return conn.execute(q).fetchone()[0]

            print(f"[diag] pipeline_events rows: {count('pipeline_events')}", flush=True)
            print(f"[diag] card_instances total: {count('card_instances')}", flush=True)
            print(f"[diag] card_instances this run: {count('card_instances', f'run_id=\"{run_id}\"')}", flush=True)
            print(f"[diag] card_views total: {count('card_views')}", flush=True)

            # Show event counts by type for this run
            if "pipeline_events" in tables:
                rows = conn.execute(
                    "SELECT event_type, COUNT(*) as n FROM pipeline_events "
                    "WHERE run_id=? GROUP BY event_type ORDER BY n DESC",
                    (run_id,),
                ).fetchall()
                for event_type, n in rows:
                    print(f"[diag]   event {event_type}: {n}", flush=True)
    except Exception as e:
        print(f"[diag] DB diagnostics error: {e}", flush=True)

    # Crops on disk
    crops_dir = output_dir / "crops"
    if crops_dir.exists():
        n_crops = len(list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png")))
        print(f"[diag] crops on disk: {n_crops}", flush=True)
    else:
        print("[diag] crops dir not found", flush=True)


runpod.serverless.start({"handler": handler})
