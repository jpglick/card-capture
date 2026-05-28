"""Lightweight performance harness for V5.5.

One command produces a JSON report comparable across branches/machines.
This phase ships the scaffold + synthetic fixture path; real-video paths
are added in later phases.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


@dataclasses.dataclass
class PerfReport:
    run_id: str
    profile: str
    video: str
    git_sha: str
    machine: Mapping[str, Any]
    timings_ms: Mapping[str, float]
    counters: Mapping[str, int]
    cards_extracted: int
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _machine_info() -> Mapping[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }


def run(profile: str, video: str, out_dir: Path) -> PerfReport:
    """Run a perf profile. Phase 0 supports the `synthetic_smoke` profile only."""
    run_id = uuid.uuid4().hex[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    counters: dict[str, int] = {}
    timings: dict[str, float] = {}
    cards = 0
    error: str | None = None
    try:
        if profile == "synthetic_smoke":
            # Phase 0: pretend stage timings; this proves the harness shape works.
            for stage in ("sample", "detect", "track", "refine", "score", "fuse", "store"):
                stage_start = time.perf_counter()
                time.sleep(0.01)
                timings[stage] = (time.perf_counter() - stage_start) * 1000.0
            counters["frames_decoded"] = 0
            counters["model_loads"] = 0
            counters["video_reopens"] = 0
            cards = 0
        elif profile == "local_v55":
            from card_capture.pipeline.request import PipelineRunRequest
            from card_capture.pipeline.runtime_local import LocalPipelineRuntime
            from card_capture.pipeline.telemetry import InMemoryTelemetry

            telemetry = InMemoryTelemetry()
            runtime = LocalPipelineRuntime(telemetry=telemetry)
            req = PipelineRunRequest(
                run_id=run_id,
                input_video=f"artifact://local/{video}",
                output_root=f"artifact://local/{out_dir / run_id}/",
                runtime_mode="cpu_debug",  # use strict_gpu when running on GPU
            )
            pipeline_start = time.perf_counter()
            result = runtime.run(req)
            timings["__pipeline__"] = (time.perf_counter() - pipeline_start) * 1000.0

            for st in result.manifest.stage_timings:
                timings[st.stage] = float(st.elapsed_ms)

            counters["frames_decoded"] = sum(
                1 for e in telemetry.events if e.payload.get("event") == "frame_decoded"
            )
            counters["model_loads"] = sum(
                1 for e in telemetry.events if e.payload.get("event") == "model_load"
            )
            counters["video_reopens"] = sum(
                1 for e in telemetry.events if e.payload.get("event") == "decode_open"
            ) - 1  # one is expected
            cards = len(result.manifest.cards)
        else:
            raise ValueError(f"unknown perf profile: {profile!r}")
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    timings["__total__"] = (time.perf_counter() - start) * 1000.0

    report = PerfReport(
        run_id=run_id,
        profile=profile,
        video=video,
        git_sha=_git_sha(),
        machine=_machine_info(),
        timings_ms=timings,
        counters=counters,
        cards_extracted=cards,
        error=error,
    )
    (out_dir / "perf_report.json").write_text(report.to_json())
    return report


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness.performance")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--profile", required=True)
    run_p.add_argument("--video", required=True)
    run_p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        report = run(args.profile, args.video, args.out)
        print(report.to_json())
        return 0 if report.error is None else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
