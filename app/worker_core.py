"""Shared GPU pipeline execution logic used by all provider workers."""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tarfile
import typing
from pathlib import Path

CUDA_CONFIG_OVERRIDES: dict = {
    "detector": "cuda",
    "device": "cuda",
    "cuda_stride": 2,
    "cuda_batch_size": 32,
    "pipeline_backend": "cuda",
}

_CONFIG_PATH = Path(__file__).parent.parent / "card_capture_config.json"

_METAFLOW_STAGE_START_RE = re.compile(
    r"\[[^/\]]+/([a-zA-Z0-9_]+)/\d+(?:\s+\([^)]+\))?\]\s+Task is starting\.?"
)


def parse_metaflow_start_stage(line: str) -> str | None:
    """Return the Metaflow step name from a task-start log line."""
    match = _METAFLOW_STAGE_START_RE.search(line)
    return match.group(1) if match else None


def apply_cuda_config() -> dict:
    """Write CUDA overrides to config; return original values for restore."""
    cfg: dict = {}
    if _CONFIG_PATH.exists():
        try:
            cfg = json.loads(_CONFIG_PATH.read_text())
        except Exception:
            pass
    original = {k: cfg.get(k) for k in CUDA_CONFIG_OVERRIDES}
    cfg.update(CUDA_CONFIG_OVERRIDES)
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return original


def restore_config(original: dict) -> None:
    """Restore config values that were overridden by apply_cuda_config."""
    if not _CONFIG_PATH.exists():
        return
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        cfg.update(original)
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


def run_pipeline(job_id: str, video_path: str, config_preset: str, output_dir: Path, stage_cb: typing.Optional[typing.Callable[[str], None]] = None) -> tuple:
    """Run the Metaflow pipeline subprocess; return (db_path, stdout).

    Streams the subprocess stdout/stderr live to our own stdout (prefixed with
    [mf]) so the RunPod worker log shows pipeline progress in real time. The
    same output is collected and returned for downstream parsing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "cards.sqlite"
    repo_root = Path(__file__).parent.parent
    cmd = [
        sys.executable, "-m", "pipeline.card_capture_flow",
        "--no-pylint", "run",
        "--video", video_path,
        "--output-dir", str(output_dir),
        "--db", str(db_path),
        "--config-preset", config_preset,
        "--ui-run-id", job_id,
        # Force the CUDA detect path. apply_cuda_config writes detector:"cuda"
        # to card_capture_config.json, but the metaflow flow reads --detector
        # from CLI args (default="docaligner"), so without this flag the
        # pipeline silently uses AdaptivePresenceSampler (CPU cv2 decode) and
        # bypasses CudaSampler entirely. Confirmed by stage_detect telemetry
        # showing sampler_type=AdaptivePresenceSampler and 117s detect with
        # only 4.7s actual YOLO time — the other 113s is CPU decode.
        "--detector", "cuda",
    ]
    env = os.environ.copy()
    # Force-set (not setdefault) so empty strings are also overridden
    if not env.get("USERNAME"):
        env["USERNAME"] = "root"
    if not env.get("USER"):
        env["USER"] = "root"
    env.setdefault("METAFLOW_USER", "runner")
    # Metaflow spawns step subprocesses that don't inherit cwd — ensure pipeline/ is importable
    existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{existing_pypath}" if existing_pypath else str(repo_root)
    env.pop("LD_PRELOAD", None)  # ensure no stale LD_PRELOAD from outer environment
    # Force unbuffered output in every Python child so lines reach us as they're
    # produced (without this, metaflow's nested subprocesses block-buffer when
    # stdout is a pipe and we see nothing until process exit).
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[diag] starting metaflow subprocess (output streamed below)", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge — we want one chronological stream
        text=True,
        bufsize=1,                  # line-buffered on our side
        cwd=str(repo_root),
        env=env,
    )
    collected: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        # Echo live to RunPod worker log so we can watch pipeline progress;
        # also collect for post-mortem timing parse and error detail.
        sys.stdout.write(f"[mf] {line}")
        sys.stdout.flush()
        collected.append(line)
        if stage_cb:
            stage = parse_metaflow_start_stage(line)
            if stage:
                stage_cb(stage)
    returncode = proc.wait()
    full_stdout = "".join(collected)

    _print_metaflow_timings(full_stdout)
    if returncode != 0:
        detail = f"STDOUT (last 4000 chars):\n{full_stdout[-4000:]}"
        raise RuntimeError(detail)
    return db_path, full_stdout


def _print_metaflow_timings(stdout: str) -> None:
    """Parse Metaflow log timestamps to produce a per-step timing table."""
    import re
    from datetime import datetime

    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
        r" \[\d+/(\w+)/\d+"
        r".*?\] (Task is starting|Task finished successfully)"
    )

    starts: dict[str, datetime] = {}
    fmt = "%Y-%m-%d %H:%M:%S.%f"

    print("[diag] === Metaflow step timings ===", flush=True)
    for ts_str, step, event in pattern.findall(stdout):
        ts = datetime.strptime(ts_str, fmt)
        if event == "Task is starting":
            starts[step] = ts
        elif event == "Task finished successfully" and step in starts:
            elapsed = (ts - starts[step]).total_seconds()
            print(f"[diag]   {step:<20} {elapsed:6.1f}s", flush=True)
    print("[diag] =================================", flush=True)


def package_results(job_id: str, output_dir: Path, db_path: Path) -> bytes:
    """Bundle crops, export.json, and worker SQLite into a gzipped tarball."""
    cards: list[dict] = []
    if db_path.exists():
        from card_capture.data.connection import read_connection
        try:
            with read_connection(db_path) as conn:
                rows = conn.execute(
                    "SELECT track_id, session_id, fused_image_path, angle"
                    " FROM card_instances WHERE run_id=?",
                    (job_id,),
                ).fetchall()
                cards = [{"track_id": r[0], "session_id": r[1],
                          "fused_image_path": r[2], "angle": r[3]} for r in rows]
        except Exception:
            pass

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        crops_dir = output_dir / "crops"
        if crops_dir.exists():
            tar.add(crops_dir, arcname="crops")
        if db_path.exists():
            tar.add(db_path, arcname="cards.sqlite")
        export_bytes = json.dumps(cards).encode()
        info = tarfile.TarInfo(name="export.json")
        info.size = len(export_bytes)
        tar.addfile(info, io.BytesIO(export_bytes))
    return buf.getvalue()
