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
from card_capture.stages import (
    store)
from card_capture.stages import dedup, fuse, resolve, score, refine, track, novelty, detect, sample
from card_capture.core.gpu_utils import probe_torch_device_status
from card_capture.core.video_utils import probe_video, _resolve_reader_backend, _decord_available, _open_capture, RollingWindowTriage, FrameTriageFilter


_STAGES = (
    ("sample", sample),
    ("detect", detect),
    ("novelty", novelty),
    ("track", track),
    ("refine", refine),
    ("score", score),
    ("resolve", resolve),
    ("fuse", fuse),
    ("dedup", dedup),
    ("store", store),
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

        # Prefer the explicit db_path callers pass in (UI, training).
        # Fall back to <output_root>/cards.sqlite for older test callers.
        if request.db_path:
            db_path = Path(request.db_path)
        else:
            output_dir_str = str(request.output_root).replace("artifact://local/", "")
            db_path = Path(output_dir_str) / "cards.sqlite"

        # Tracker inference is guarded here. Refinement GPU-boundary decomposition is
        # separate architectural debt; do not add new main-thread Torch calls.
        from card_capture.pipeline.runtime_worker import RuntimeWorker
        runtime_worker = RuntimeWorker()
        writer = Writer(db_path)

        # State carried across stages — frames, detections, tracks, crops, scores, etc.
        # The actual shape grows as Tasks 3.3-3.8 wire stages.
        state: dict = {
            "request": request,
            "video_id": request.video_id or 0,
            "config_preset": request.config_preset,
            "db_path": db_path,
            "runtime_worker": runtime_worker,
            # Phase 4 wiring — inject repositories and output_root Path
            "output_root": Path(str(request.output_root).replace("artifact://local/", "")),
            "repos": {
                "runs": RunsRepository(writer, db_path),
                "events": EventsRepository(writer, db_path),
                "cards": CardsRepository(writer, db_path),
            }
        }

        try:
            runtime_worker.start()
            writer.start()

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
                stage_metrics = state.get("stage_metrics", {}).get(name, {})
                self._telemetry.stage_finished(name, elapsed_ms, stage_metrics)
        finally:
            runtime_worker.stop()
            writer.stop()

        manifest = RunManifest(
            run_id=run_id,
            runtime_mode=request.runtime_mode,
            input_video=request.input_video,
            output_artifacts=state.get("output_artifacts", []),
            cards=state.get("cards", []),
            stage_timings=timings,
            contract_violations=violations,
            version="0.5.5+phase4",
        )
        return PipelineRunResult(manifest=manifest)
