"""Shared GPU pipeline execution logic used by all provider workers."""
from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

CUDA_CONFIG_OVERRIDES: dict = {
    "detector": "cuda",
    "device": "cuda",
    "cuda_stride": 2,
    "cuda_batch_size": 32,
    "pipeline_backend": "cuda",
}

_CONFIG_PATH = Path(__file__).parent.parent / "card_capture_config.json"


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


def run_pipeline(job_id: str, video_path: str, config_preset: str, output_dir: Path) -> tuple:
    """Run the Metaflow pipeline subprocess; return (db_path, stdout)."""
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
    # GPU-compiled decord needs GLIBCXX_3.4.30 which conda's /opt/conda/lib/libstdc++.so.6 lacks.
    # LD_PRELOAD had side-effects (broke cuvidDestroyDecoder symbol resolution).
    # LD_LIBRARY_PATH works if conda uses DT_RUNPATH (it does — conda builds use RUNPATH).
    # Put system libs first so the dynamic linker finds the right libstdc++ before conda's.
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"/usr/lib/x86_64-linux-gnu:{existing_ld}".rstrip(":")
    env.pop("LD_PRELOAD", None)  # remove if set from outer environment
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), env=env)
    _print_metaflow_timings(result.stdout)
    if result.returncode != 0:
        detail = f"STDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        raise RuntimeError(detail)
    return db_path, result.stdout


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
    """Bundle crops + export.json into a gzipped tarball; return as bytes."""
    cards: list[dict] = []
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT track_id, session_id, fused_image_path, angle"
                    " FROM card_instances WHERE run_id=?",
                    (job_id,),
                ).fetchall()
                cards = [dict(r) for r in rows]
        except Exception:
            pass

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        crops_dir = output_dir / "crops"
        if crops_dir.exists():
            tar.add(crops_dir, arcname="crops")
        export_bytes = json.dumps(cards).encode()
        info = tarfile.TarInfo(name="export.json")
        info.size = len(export_bytes)
        tar.addfile(info, io.BytesIO(export_bytes))
    return buf.getvalue()
