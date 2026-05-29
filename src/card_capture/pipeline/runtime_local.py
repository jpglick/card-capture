"""LocalPipelineRuntime: executes all stages in one process.

This is the V5.5 replacement for the Metaflow flow. Stages run as direct
function calls; loaded models, decoded frames, and GPU-resident tensors
are passed between stages as in-memory objects.

The runtime selects an execution backend (StrictGpu / CpuDebug) based on
`request.runtime_mode`. Backend-specific decode and model loading live in
the backend modules; this orchestrator owns sequencing, telemetry, and
manifest construction.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
    StageTiming,
)
from card_capture.pipeline.telemetry import PipelineTelemetry, NoopTelemetry
from card_capture.pipeline.stages import (
    sample as stage_sample,
    detect as stage_detect,
    novelty as stage_novelty,
    track as stage_track,
    refine as stage_refine,
    score as stage_score,
    resolve as stage_resolve,
    fuse as stage_fuse,
    dedup as stage_dedup,
    store as stage_store,
)


_STAGES = (
    ("sample", stage_sample),
    ("detect", stage_detect),
    ("novelty", stage_novelty),
    ("track", stage_track),
    ("refine", stage_refine),
    ("score", stage_score),
    ("resolve", stage_resolve),
    ("fuse", stage_fuse),
    ("dedup", stage_dedup),
    ("store", stage_store),
)


from card_capture.pipeline.runner import PipelineRunHandle


class LocalPipelineRuntime:
    def __init__(self, telemetry: PipelineTelemetry | None = None) -> None:
        self._telemetry = telemetry or NoopTelemetry()

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        """Synchronously execute the run."""
        # For LocalPipelineRuntime, submit() just runs it and returns a handle
        # that wait() can then use to return the result.
        self._last_result = self.run(request)
        return PipelineRunHandle(run_id=request.run_id, backend="local")

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        """Return the result of the last run."""
        # This implementation is toy-like because it's synchronous;
        # the result is already there.
        if not hasattr(self, "_last_result"):
            raise RuntimeError("Must call submit() before wait() on LocalPipelineRuntime")
        return self._last_result

    def cancel(self, handle: PipelineRunHandle) -> None:
        """No-op for synchronous local runtime."""
        pass

    def run(self, request: PipelineRunRequest) -> PipelineRunResult:
        timings: list[StageTiming] = []
        violations: list = []
        run_id = request.run_id or uuid.uuid4().hex[:12]
        
        # Initialize DAL
        from card_capture.data.writer import Writer
        from card_capture.data.repositories.runs import RunsRepository
        from card_capture.data.repositories.events import EventsRepository
        from card_capture.data.repositories.cards import CardsRepository
        
        # We need to extract the path without artifact:// prefix for local files
        db_path_str = str(request.output_root).replace("artifact://local/", "")
        db_path = Path(db_path_str) / "cards.sqlite" # Assuming db is in output_root or passed explicitly
        # Wait, the spec says run_context has db_path, but PipelineRunRequest just has output_root.
        # Actually in CLI I passed req = PipelineRunRequest(output_root=f"artifact://local/{args.output_dir}/")
        # And the db was passed to the storage earlier.
        
        writer = Writer(db_path)
        writer.start()

        # State carried across stages — frames, detections, tracks, crops, scores, etc.
        # The actual shape grows as Tasks 3.3-3.8 wire stages.
        state: dict = {
            "request": request,
            "repos": {
                "runs": RunsRepository(writer, db_path),
                "events": EventsRepository(writer, db_path),
                "cards": CardsRepository(writer, db_path),
            }
        }

        try:
            for name, module in _STAGES:
                self._telemetry.stage_started(name, {})
                start = time.perf_counter()
                try:
                    module.run(state, telemetry=self._telemetry)
                except Exception as exc:
                    violations.append({"code": f"stage_failed:{name}", "metadata": {"error": repr(exc)}})
                    self._telemetry.contract_violation(
                        f"stage_failed:{name}", {"error": repr(exc)}
                    )
                    raise
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                timings.append(StageTiming(stage=name, elapsed_ms=elapsed_ms))
                self._telemetry.stage_finished(name, elapsed_ms, {})
        finally:
            writer.stop()

        manifest = RunManifest(
            run_id=run_id,
            runtime_mode=request.runtime_mode,
            input_video=request.input_video,
            output_artifacts=state.get("output_artifacts", []),
            cards=state.get("cards", []),
            stage_timings=timings,
            contract_violations=violations,
            version="0.5.5+phase3",
        )
        return PipelineRunResult(manifest=manifest)
