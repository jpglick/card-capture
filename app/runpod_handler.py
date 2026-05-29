"""RunPod serverless handler — deployed to RunPod, not run locally.

Deploy by building a Docker image with this file as the entrypoint:
    CMD ["python", "-m", "app.runpod_handler"]

R2 credentials are injected as RunPod endpoint environment variables:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
"""
from __future__ import annotations

import os
import subprocess as _sp
import time
from pathlib import Path

import boto3
import runpod
from botocore.config import Config as BotocoreConfig

from app.worker_core import apply_cuda_config, restore_config, run_pipeline, package_results
from card_capture.data.sql_queries import (
    RUNPOD_HANDLER_EVENTS_BY_TYPE,
    RUNPOD_HANDLER_PIPELINE_EVENTS_PRAGMA,
    RUNPOD_HANDLER_RUN_TELEMETRY,
    RUNPOD_HANDLER_STAGE_EVENTS,
    RUNPOD_HANDLER_TABLES,
    runpod_handler_count_query,
)

_NVDEC_PROBE_VIDEO = Path("/tmp/cc_nvdec_preflight.mp4")
_NVDEC_PROBE_MIN_BYTES = 4096


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
    _assert_gpu_preflight(gpu_info)

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
    db_path = output_dir / "cards.sqlite"  # deterministic; valid to read on failure

    # Start background resource sampler so we know peak GPU/VRAM/decoder/CPU/RAM
    # during the pipeline — needed to make informed batch_size / stride tuning
    # decisions instead of guessing. The single _check_gpu() snapshot above
    # only sees torch's idle state (e.g. 506 MB VRAM) and tells us nothing.
    sampler = _ResourceSampler(interval_s=0.5)
    sampler.start()

    original = apply_cuda_config()
    pipeline_stdout = ""
    try:
        _, pipeline_stdout = run_pipeline(run_id, str(video_path), config_preset, output_dir, stage_cb=sampler.set_stage)
    except Exception:
        print("[diag] pipeline failed — collecting partial DB diagnostics", flush=True)
        try:
            _collect_db_diagnostics(run_id, db_path, output_dir)
        except Exception as diag_e:
            print(f"[diag] diagnostics collection failed: {diag_e}", flush=True)
        raise
    finally:
        restore_config(original)
        sampler.stop()
    t_pipeline = time.time() - t0
    t0 = time.time()

    resource_stats = sampler.summary()
    print(f"[diag] resource_stats: {resource_stats}", flush=True)

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
        "resource_stats": resource_stats,
        "resource_samples": sampler.samples(),
        "metaflow_stdout_tail": pipeline_stdout[-200_000:],
        "timings": {
            "r2_download_s": round(t_download, 1),
            "pipeline_s": round(t_pipeline, 1),
            "r2_upload_out_s": round(t_upload_out, 1),
            "total_s": round(total, 1),
            "steps": step_timings,
        },
        "diagnostics": db_diag,
    }


class _ResourceSampler:
    """Background sampler — nvidia-smi + /proc/stat + /proc/meminfo every interval_s.

    Tracks peak + mean for GPU util, NVDEC decoder util, NVENC encoder util,
    VRAM used (MB), CPU% (system), RAM used (MB). Surfaces enough to answer:
    "are we VRAM-bound? compute-bound? could we batch larger?"
    """
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._stop = False
        self._thread = None
        self._samples: list[dict] = []
        self.current_stage = "init"

    def set_stage(self, stage: str) -> None:
        self.current_stage = stage

    def start(self) -> None:
        import threading
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        import subprocess as _sp
        # Try to import psutil; fall back to /proc parsing if missing
        try:
            import psutil as _psutil
            have_psutil = True
        except Exception:
            have_psutil = False
        while not self._stop:
            sample: dict = {"ts": time.time(), "stage": self.current_stage}
            # nvidia-smi: gpu util, decoder util, encoder util, vram used MB
            try:
                r = _sp.run([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,utilization.decoder,utilization.encoder,memory.used",
                    "--format=csv,noheader,nounits"
                ], capture_output=True, text=True, timeout=3)
                if r.returncode == 0:
                    parts = [p.strip() for p in r.stdout.strip().split(",")]
                    if len(parts) >= 5:
                        sample["gpu_pct"] = int(parts[0])
                        sample["mem_io_pct"] = int(parts[1])
                        sample["decoder_pct"] = int(parts[2])
                        sample["encoder_pct"] = int(parts[3])
                        sample["vram_used_mb"] = int(parts[4])
            except Exception:
                pass
            # System CPU + RAM
            if have_psutil:
                try:
                    sample["cpu_pct"] = _psutil.cpu_percent(interval=None)
                    vm = _psutil.virtual_memory()
                    sample["ram_used_mb"] = int(vm.used / 1_048_576)
                    sample["ram_pct"] = vm.percent
                except Exception:
                    pass
            self._samples.append(sample)
            time.sleep(self.interval_s)

    def samples(self) -> list[dict]:
        if not self._samples:
            return []
        t0 = self._samples[0]["ts"]
        result = []
        for sample in self._samples:
            row = {k: v for k, v in sample.items() if k != "ts"}
            row["elapsed_s"] = round(sample["ts"] - t0, 3)
            result.append(row)
        return result

    def summary(self) -> dict:
        if not self._samples:
            return {"error": "no samples collected"}

        def _stats(key: str) -> dict | None:
            vals = [s[key] for s in self._samples if key in s]
            if not vals: return None
            return {"peak": max(vals), "mean": round(sum(vals) / len(vals), 1), "n": len(vals)}

        stage_peaks = {}
        for s in self._samples:
            stage = s.get("stage", "unknown")
            vram = s.get("vram_used_mb", 0)
            if vram > stage_peaks.get(stage, 0):
                stage_peaks[stage] = vram

        return {
            "samples": len(self._samples),
            "interval_s": self.interval_s,
            "gpu_pct":      _stats("gpu_pct"),
            "decoder_pct":  _stats("decoder_pct"),
            "encoder_pct":  _stats("encoder_pct"),
            "mem_io_pct":   _stats("mem_io_pct"),
            "vram_used_mb": _stats("vram_used_mb"),
            "cpu_pct":      _stats("cpu_pct"),
            "ram_used_mb":  _stats("ram_used_mb"),
            "ram_pct":      _stats("ram_pct"),
            "vram_peak_by_stage": stage_peaks,
        }


def _collect_db_diagnostics(run_id: str, db_path: Path, output_dir: Path) -> dict:
    """Collect card/frame counts from the pipeline DB and return as dict."""
    from card_capture.data.connection import read_connection
    result: dict = {}
    if not db_path.exists():
        result["error"] = "DB not found"
        return result
    try:
        with read_connection(db_path) as conn:
            tables = {r[0] for r in conn.execute(RUNPOD_HANDLER_TABLES).fetchall()}

            def count(table: str, where: str = "") -> int:
                if table not in tables:
                    return -1
                q = runpod_handler_count_query(table, where)
                return conn.execute(q).fetchone()[0]

            result["card_instances_total"] = count("card_instances")
            result["card_instances_this_run"] = count("card_instances", 'run_id="' + run_id + '"')
            result["card_views_total"] = count("card_views")
            result["pipeline_events_total"] = count("pipeline_events")

            if "pipeline_events" in tables:
                rows = conn.execute(RUNPOD_HANDLER_EVENTS_BY_TYPE, (run_id,)).fetchall()
                result["events"] = {et: n for et, n in rows}

                # Surface stage timings + any payload fields (event_data is JSON
                # blob per stage with frame counts, device, durations, etc.)
                cols = [c[1] for c in conn.execute(RUNPOD_HANDLER_PIPELINE_EVENTS_PRAGMA).fetchall()]
                if "data_json" in cols:
                    stage_rows = conn.execute(RUNPOD_HANDLER_STAGE_EVENTS, (run_id,)).fetchall()
                    import json as _json
                    stage_payloads = {}
                    for et, blob in stage_rows:
                        if blob:
                            try:
                                stage_payloads[et] = _json.loads(blob)
                            except Exception:
                                stage_payloads[et] = {"raw": str(blob)[:500]}
                    if stage_payloads:
                        result["stage_payloads"] = stage_payloads

            # detect_telemetry_json answers: was YOLO on cuda? how many batches?
            # how many frames hit YOLO? triage pass rate? — all the slow-step questions.
            if "pipeline_runs" in tables:
                row = conn.execute(RUNPOD_HANDLER_RUN_TELEMETRY, (run_id,)).fetchone()
                if row and row[0]:
                    import json as _json
                    try:
                        result["detect_telemetry"] = _json.loads(row[0])
                    except Exception:
                        result["detect_telemetry"] = {"raw": row[0]}
                else:
                    result["detect_telemetry"] = "pipeline_runs row missing or empty for this run_id"
            else:
                result["detect_telemetry"] = "pipeline_runs table does not exist"
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


def _find_libnvcuvid() -> str | None:
    """Return a real NVDEC driver library path if one is mounted."""
    import ctypes.util

    for candidate in (
        "/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvcuvid.so",
        "/usr/local/nvidia/lib64/libnvcuvid.so.1",
        "/usr/local/nvidia/lib64/libnvcuvid.so",
    ):
        if Path(candidate).exists():
            return candidate

    found = ctypes.util.find_library("nvcuvid")
    return found if found else None


def _video_device_nodes() -> list[str]:
    nodes = []
    for candidate in ("/dev/nvidia-modeset", "/dev/nvidia-caps", "/dev/dri"):
        if Path(candidate).exists():
            nodes.append(candidate)
    return nodes


def _ensure_nvdec_probe_video() -> Path:
    path = _NVDEC_PROBE_VIDEO
    if path.exists() and path.stat().st_size >= _NVDEC_PROBE_MIN_BYTES:
        return path
    path.unlink(missing_ok=True)

    result = _sp.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=128x128:rate=30:duration=2",
            "-frames:v",
            "60",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
            "-y",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg probe video generation failed")
    return path


def _check_nvdec_video_capability() -> dict:
    info: dict = {
        "libnvcuvid_present": False,
        "libnvcuvid_path": None,
        "video_device_nodes": _video_device_nodes(),
        "decord_gpu_context": False,
        "decord_nvdec_probe": False,
    }

    lib_path = _find_libnvcuvid()
    info["libnvcuvid_path"] = lib_path
    info["libnvcuvid_present"] = bool(lib_path)
    if not lib_path:
        info["error"] = "libnvcuvid.so.1 not found in container runtime"
        return info

    try:
        import decord

        ctx = decord.gpu(0)
        info["decord_gpu_context"] = True
        probe_video = _ensure_nvdec_probe_video()
        vr = decord.VideoReader(str(probe_video), ctx=ctx)
        batch = vr.get_batch([0])
        # Force materialization so decoder startup failures surface here.
        if hasattr(batch, "asnumpy"):
            batch.asnumpy()
        elif hasattr(batch, "cpu"):
            batch.cpu().numpy()
        info["decord_nvdec_probe"] = True
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _assert_gpu_preflight(info: dict) -> None:
    capabilities = info.get("capabilities", {})
    missing = [name for name in ("compute", "utility", "video") if not capabilities.get(name)]
    if not missing:
        return
    detail = info.get("video", {}).get("error") or info.get("error") or info
    raise RuntimeError(
        "RunPod GPU preflight failed; missing capabilities: "
        f"{', '.join(missing)}. Details: {detail}"
    )


def _check_gpu() -> dict:
    try:
        import torch
        info: dict = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "nvidia_driver_capabilities_env": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", ""),
        }
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            mem = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(mem / 1e9, 1)
        utility_ok = False
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
                utility_ok = True
            else:
                info["nvidia_smi_error"] = r.stderr.strip()
        except Exception:
            info["nvidia_smi_error"] = "nvidia-smi failed"

        video_info = _check_nvdec_video_capability()
        info["video"] = video_info
        info["capabilities"] = {
            "compute": bool(info["cuda_available"] and info["device_count"] > 0),
            "utility": utility_ok,
            "video": bool(
                video_info.get("libnvcuvid_present")
                and video_info.get("decord_gpu_context")
                and video_info.get("decord_nvdec_probe")
            ),
        }
        return info
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
