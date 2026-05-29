from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.pipeline.request import PipelineRunRequest, PipelineRunResult, RunManifest
from card_capture.pipeline.runner import PipelineRunHandle
from card_capture.platforms.local import LocalRunner


class _StubRuntime:
    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        self._result = PipelineRunResult(
            manifest=RunManifest(
                run_id=request.run_id,
                runtime_mode=request.runtime_mode,
                input_video=request.input_video,
                output_artifacts=[],
                cards=[],
                stage_timings=[],
                contract_violations=[],
                version="test",
            )
        )
        return PipelineRunHandle(run_id=request.run_id, backend="local")

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        return self._result

    def cancel(self, handle: PipelineRunHandle) -> None:
        return


def test_submit_then_wait_returns_manifest(tmp_path: Path) -> None:
    runner = LocalRunner(runtime=_StubRuntime())
    req = PipelineRunRequest(
        run_id="local-1",
        input_video=f"artifact://local/{tmp_path}/fake.mov",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
    )
    handle = runner.submit(req)
    assert handle.backend == "local"
    assert handle.run_id == "local-1"
    result = runner.wait(handle)
    assert result.manifest.run_id == "local-1"


def test_cancel_is_idempotent_noop() -> None:
    runner = LocalRunner()
    runner.cancel(PipelineRunHandle(run_id="x", backend="local"))
    runner.cancel(PipelineRunHandle(run_id="x", backend="local"))


def test_wait_on_unknown_handle_raises() -> None:
    runner = LocalRunner()
    with pytest.raises(KeyError):
        runner.wait(PipelineRunHandle(run_id="nope", backend="beam"))
