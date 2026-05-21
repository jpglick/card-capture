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

    # ── GPU check ────────────────────────────────────────────────────────────
    gpu_info = _check_gpu()
    print(f"[diag] GPU: {gpu_info}", flush=True)

    s3 = _r2_client()

    # ── Video download ───────────────────────────────────────────────────────
    video_path = Path(f"/tmp/{run_id}_input.mov")
    s3.download_file(bucket, video_key, str(video_path))
    video_mb = video_path.stat().st_size / 1_048_576
    t_download = time.time() - t0
    print(f"[diag] R2 download ({video_mb:.1f} MB): {t_download:.1f}s", flush=True)
    t0 = time.time()

    # ── Pipeline ─────────────────────────────────────────────────────────────
    output_dir = Path(f"/tmp/cc_output/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = apply_cuda_config()
    pipeline_stdout = ""
    try:
        db_path, pipeline_stdout = run_pipeline(run_id, str(video_path), config_preset, output_dir)
    finally:
        restore_config(original)
    t_pipeline = time.time() - t0
    t0 = time.time()

    # ── Diagnostics from DB ──────────────────────────────────────────────────
    db_diag = _collect_db_diagnostics(run_id, db_path, output_dir)

    # ── Step timings from Metaflow stdout ────────────────────────────────────
    step_timings = _parse_metaflow_timings(pipeline_stdout)

    # ── Package ──────────────────────────────────────────────────────────────
    tarball_bytes = package_results(run_id, output_dir, db_path)
    tarball_mb = len(tarball_bytes) / 1_048_576
    t0 = _t(f"package results ({tarball_mb:.1f} MB tarball)", t0)

    # ── Upload ───────────────────────────────────────────────────────────────
    s3.put_object(Bucket=bucket, Key=results_key, Body=tarball_bytes)
    t_upload_out = time.time() - t0

    total = time.time() - job_start
    print(f"[diag] TOTAL handler time: {total:.1f}s", flush=True)

    return {
        "status": "complete",
        "results_key": results_key,
        "gpu": gpu_info,
        "timings": {
            "r2_download_s": round(t_download, 1),
            "pipeline_s": round(t_pipeline, 1),
            "r2_upload_out_s": round(t_upload_out, 1),
            "total_s": round(total, 1),
            "steps": step_timings,
        },
        "diagnostics": db_diag,
    }


def _collect_db_diagnostics(run_id: str, db_path: Path, output_dir: Path) -> dict:
    """Collect card/frame counts from the pipeline DB and return as dict."""
    import sqlite3 as _sqlite3
    result: dict = {}
    if not db_path.exists():
        result["error"] = "DB not found"
        return result
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

            result["card_instances_total"] = count("card_instances")
            result["card_instances_this_run"] = count("card_instances", 'run_id="' + run_id + '"')
            result["card_views_total"] = count("card_views")
            result["pipeline_events_total"] = count("pipeline_events")

            if "pipeline_events" in tables:
                rows = conn.execute(
                    "SELECT event_type, COUNT(*) as n FROM pipeline_events "
                    "WHERE run_id=? GROUP BY event_type ORDER BY n DESC",
                    (run_id,),
                ).fetchall()
                result["events"] = {et: n for et, n in rows}
    except Exception as e:
        result["error"] = str(e)

    crops_dir = output_dir / "crops"
    result["crops_on_disk"] = len(list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png"))) if crops_dir.exists() else 0
    for k, v in result.items():
        print(f"[diag] {k}: {v}", flush=True)
    return result


def _parse_metaflow_timings(stdout: str) -> dict:
    """Parse Metaflow log timestamps to produce a per-step timing dict."""
    import re
    from datetime import datetime

    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
        r" \[\d+/(\w+)/\d+"
        r".*?\] (Task is starting|Task finished successfully)"
    )
    starts: dict = {}
    timings: dict = {}
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    for ts_str, step, event in pattern.findall(stdout):
        ts = datetime.strptime(ts_str, fmt)
        if event == "Task is starting":
            starts[step] = ts
        elif event == "Task finished successfully" and step in starts:
            timings[step] = round((ts - starts[step]).total_seconds(), 1)
    print(f"[diag] step timings: {timings}", flush=True)
    return timings


def _check_gpu() -> dict:
    try:
        import torch
        import subprocess as _sp
        info: dict = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            mem = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(mem / 1e9, 1)
        # Also try nvidia-smi for utilization
        try:
            r = _sp.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                parts = r.stdout.strip().split(", ")
                info["gpu_util_pct"] = parts[0]
                info["vram_used_mb"] = parts[1]
                info["vram_total_mb"] = parts[2]
        except Exception:
            pass
        return info
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
