from __future__ import annotations

from card_capture.pipeline.request import PipelineRunRequest, RunManifest
from card_capture.platforms.beam import BeamRunner, BeamRunnerError


class _StubEndpoint:
    def __init__(self) -> None:
        self._jobs: dict[str, _StubJob] = {}

    def run(self, payload: dict) -> "_StubJob":
        job = _StubJob(run_id=payload["run_id"])
        self._jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> "_StubJob":
        return self._jobs[job_id]


class _StubJob:
    def __init__(self, run_id: str, force_status: str | None = None, error_msg: str = "") -> None:
        self.id = f"bm-{run_id}"
        self._run_id = run_id
        self._status = force_status or "IN_PROGRESS"
        self._error = error_msg

    def status(self) -> str:
        if self._status == "IN_PROGRESS":
            self._status = "COMPLETED"
        return self._status

    def output(self) -> dict:
        manifest = RunManifest(
            run_id=self._run_id,
            runtime_mode="strict_gpu",
            input_video="artifact://s3/x.MOV",
            output_artifacts=[],
            cards=[],
            stage_timings=[],
            contract_violations=[],
            version="0.5.5+test",
        )
        return {"manifest_json": manifest.to_json()}

    def cancel(self) -> None:
        self._status = "CANCELLED"

    def error(self) -> str:
        return self._error


def test_submit_wait_roundtrip() -> None:
    endpoint = _StubEndpoint()
    runner = BeamRunner(endpoint=endpoint, poll_interval=0.0)
    req = PipelineRunRequest(
        run_id="bm1",
        input_video="artifact://s3/x.MOV",
        output_root="artifact://s3/bm1/",
        runtime_mode="strict_gpu",
    )
    handle = runner.submit(req)
    assert handle.backend == "beam"
    assert handle.opaque == "bm-bm1"
    result = runner.wait(handle)
    assert result.manifest.run_id == "bm1"


def test_failed_job_raises_categorized_error() -> None:
    endpoint = _StubEndpoint()
    failed = _StubJob(run_id="bm2", force_status="FAILED", error_msg="task failed during exec")
    endpoint._jobs[failed.id] = failed
    runner = BeamRunner(endpoint=endpoint, poll_interval=0.0)
    import pytest as _pytest

    with _pytest.raises(BeamRunnerError) as exc_info:
        runner.wait(type("H", (), {"opaque": failed.id, "run_id": "bm2", "backend": "beam"})())
    assert exc_info.value.failure.category == "execution_failed"
