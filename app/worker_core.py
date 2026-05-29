"""Shared GPU pipeline execution logic used by all provider workers."""
from __future__ import annotations

import io
import json
import time
import tarfile
import typing
from pathlib import Path
from card_capture.data.sql_queries import WORKER_CORE_CARD_EXPORT

CUDA_CONFIG_OVERRIDES: dict = {
    "detector": "cuda",
    "device": "cuda",
    "cuda_stride": 2,
    "cuda_batch_size": 32,
    "pipeline_backend": "cuda",
}

_CONFIG_PATH = Path(__file__).parent.parent / "card_capture_config.json"


def parse_metaflow_start_stage(line: str) -> str | None:
    """Legacy shim — Metaflow log scraping is gone.

    Kept so old imports still resolve; always returns ``None`` now that we
    no longer subprocess Metaflow. Callers should switch to the per-stage
    ``stage_cb`` plumbed through :func:`run_pipeline`.
    """
    return None


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


class _WorkerCoreTelemetry:
    """Minimal PipelineTelemetry that prints stage lines + drives stage_cb.

    Mirrors what the metaflow stdout scraper used to do: tell RunPod workers
    which stage we're in (so they can update status) and emit a timing table
    at the end.
    """

    def __init__(self, stage_cb: typing.Optional[typing.Callable[[str], None]]) -> None:
        self._stage_cb = stage_cb
        self.timings: dict[str, float] = {}

    def stage_started(self, stage: str, metadata) -> None:
        print(f"[unified] stage {stage} started", flush=True)
        if self._stage_cb:
            try:
                self._stage_cb(stage)
            except Exception as exc:
                print(f"[unified] stage_cb failed for {stage}: {exc}", flush=True)

    def stage_finished(self, stage: str, elapsed_ms: int, metadata) -> None:
        self.timings[stage] = elapsed_ms / 1000.0
        print(f"[unified] stage {stage} finished in {elapsed_ms}ms", flush=True)

    def resource_sample(self, sample) -> None:
        pass

    def contract_violation(self, code: str, metadata) -> None:
        print(f"[unified] contract violation: {code} {dict(metadata)}", flush=True)


def run_pipeline(job_id: str, video_path: str, config_preset: str, output_dir: Path, stage_cb: typing.Optional[typing.Callable[[str], None]] = None) -> tuple:
    """Run the unified in-process pipeline; return (db_path, stdout).

    Post-v5.5 replacement for the Metaflow subprocess. Stages run in this
    process via :class:`LocalPipelineRuntime`; ``stage_cb`` still fires once
    per stage start so RunPod can publish status updates.

    The returned ``stdout`` is a synthesized summary (manifest JSON +
    per-stage timing table) so existing callers that pattern-match against
    the legacy Metaflow output keep working at the surface level.
    """
    from card_capture.pipeline.request import PipelineRunRequest
    from card_capture.pipeline.runtime_local import LocalPipelineRuntime

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "cards.sqlite"

    telemetry = _WorkerCoreTelemetry(stage_cb)
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    request = PipelineRunRequest(
        run_id=job_id,
        input_video=f"artifact://local/{video_path}",
        output_root=f"artifact://local/{output_dir.resolve()}/",
        # apply_cuda_config() flips detector/device in card_capture_config.json
        # which the stages still read at construction time; this request stays
        # strict_gpu so the runtime knows it's allowed to touch CUDA.
        runtime_mode="strict_gpu",
        config={"detector": "cuda", "device": "cuda"},
        db_path=str(db_path.resolve()),
        config_preset=config_preset,
    )

    print(f"[diag] starting in-process unified runtime", flush=True)
    started = time.monotonic()
    try:
        result = runtime.run(request)
    except Exception as exc:
        raise RuntimeError(f"Unified pipeline failed: {exc!r}") from exc
    elapsed_total = time.monotonic() - started

    if result.manifest.contract_violations:
        raise RuntimeError(
            f"Pipeline contract violations: {result.manifest.contract_violations}"
        )

    # Build a faux-stdout for legacy callers + log the timing table.
    print("[diag] === Unified runtime stage timings ===", flush=True)
    for stage, secs in telemetry.timings.items():
        print(f"[diag]   {stage:<20} {secs:6.1f}s", flush=True)
    print(f"[diag]   total{'':<16}{elapsed_total:6.1f}s", flush=True)
    print("[diag] ====================================", flush=True)

    summary_lines = [f"manifest: {result.manifest.to_json()}"]
    summary_lines += [f"timing {s}: {t:.2f}s" for s, t in telemetry.timings.items()]
    summary_lines.append(f"timing total: {elapsed_total:.2f}s")
    full_stdout = "\n".join(summary_lines) + "\n"
    return db_path, full_stdout


def package_results(job_id: str, output_dir: Path, db_path: Path) -> bytes:
    """Bundle crops, export.json, and worker SQLite into a gzipped tarball."""
    cards: list[dict] = []
    if db_path.exists():
        from card_capture.data.connection import read_connection
        try:
            with read_connection(db_path) as conn:
                rows = conn.execute(
                    WORKER_CORE_CARD_EXPORT,
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
