"""Strict GPU runtime wrapper.

In production this runtime does NOT globally monkeypatch torch. Instead, it
exposes only the safe device-tagged batch APIs to stage code. Forbidden
imports inside strict stage modules are caught statically by the GPU-strict
AST scanner (Phase 2 blocking).

Set CC_GPU_STRICT=1 to enable additional runtime assertion checks (device
tags, batch invariants) — this does NOT enable global monkeypatching.
"""
from __future__ import annotations

import os
import time

import torch

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
    StageTiming,
)
from card_capture.pipeline.telemetry import PipelineTelemetry, NoopTelemetry
from card_capture.runtime.gpu_session import GpuSession, MissingGpuError


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise MissingGpuError("No CUDA or MPS device available for StrictGpuRuntime")


class StrictGpuRuntime:
    def __init__(self, telemetry: PipelineTelemetry | None = None) -> None:
        self._telemetry = telemetry or NoopTelemetry()
        self._assert_mode = os.environ.get("CC_GPU_STRICT") == "1"

    def run(self, request: PipelineRunRequest) -> PipelineRunResult:
        if request.runtime_mode != "strict_gpu":
            raise ValueError(
                f"StrictGpuRuntime requires runtime_mode='strict_gpu', got {request.runtime_mode!r}"
            )
        device = _select_device()
        session = GpuSession(device=device, strict=True, telemetry=self._telemetry)

        start = time.perf_counter()
        timings: list[StageTiming] = []
        # Stage wiring lands in Phase 3.
        _ = session  # silence unused while skeleton

        manifest = RunManifest(
            run_id=request.run_id,
            runtime_mode="strict_gpu",
            input_video=request.input_video,
            output_artifacts=[],
            cards=[],
            stage_timings=timings,
            contract_violations=[],
            version="0.5.5+phase2",
            metadata={"phase": "phase2-skeleton", "device": str(device)},
        )
        self._telemetry.stage_finished(
            "__total__", int((time.perf_counter() - start) * 1000), {"device": str(device)}
        )
        return PipelineRunResult(manifest=manifest)
