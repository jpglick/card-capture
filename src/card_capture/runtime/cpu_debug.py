"""CPU debug runtime.

A minimal, intentionally slow implementation of the PipelineRuntime contract
that runs on CPU only. Used for deterministic local debugging and CI.

Stage execution is wired in Phase 3 when stage facades exist. For now,
the runtime returns an empty manifest so the contract is satisfied.
"""
from __future__ import annotations

import time

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
    StageTiming,
)
from card_capture.pipeline.telemetry import PipelineTelemetry, NoopTelemetry


class CpuDebugRuntime:
    def __init__(self, telemetry: PipelineTelemetry | None = None) -> None:
        self._telemetry = telemetry or NoopTelemetry()

    def run(self, request: PipelineRunRequest) -> PipelineRunResult:
        if request.runtime_mode != "cpu_debug":
            raise ValueError(
                f"CpuDebugRuntime requires runtime_mode='cpu_debug', got {request.runtime_mode!r}"
            )
        start = time.perf_counter()
        # Stage wiring lands in Phase 3. Phase 2: return a well-formed manifest.
        timings: list[StageTiming] = []
        manifest = RunManifest(
            run_id=request.run_id,
            runtime_mode="cpu_debug",
            input_video=request.input_video,
            output_artifacts=[],
            cards=[],
            stage_timings=timings,
            contract_violations=[],
            version="0.5.5+phase2",
            metadata={"phase": "phase2-skeleton"},
        )
        self._telemetry.stage_finished(
            "__total__", int((time.perf_counter() - start) * 1000), {}
        )
        return PipelineRunResult(manifest=manifest)
