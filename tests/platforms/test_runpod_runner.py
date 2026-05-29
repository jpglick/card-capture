from __future__ import annotations

from card_capture.pipeline.request import PipelineRunRequest, RunManifest
from card_capture.platforms.runpod import RunpodRunner, RunpodRunnerError


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
        self.id = f"rp-{run_id}"
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


def _make_runner(endpoint) -> RunpodRunner:
    r = RunpodRunner.__new__(RunpodRunner)
    r._endpoint = endpoint
    r._poll_interval = 0.0
    return r


def test_submit_wait_roundtrip() -> None:
    endpoint = _StubEndpoint()
    runner = _make_runner(endpoint)
    req = PipelineRunRequest(
        run_id="rp1",
        input_video="artifact://s3/x.MOV",
        output_root="artifact://s3/rp1/",
        runtime_mode="strict_gpu",
    )
    handle = runner.submit(req)
    assert handle.backend == "runpod"
    assert handle.opaque == "rp-rp1"
    result = runner.wait(handle)
    assert result.manifest.run_id == "rp1"


def test_failed_job_raises_categorized_error() -> None:
    endpoint = _StubEndpoint()
    failed = _StubJob(run_id="rp2", force_status="FAILED", error_msg="JOB FAILED: out of memory")
    endpoint._jobs[failed.id] = failed
    runner = _make_runner(endpoint)
    import pytest as _pytest

    with _pytest.raises(RunpodRunnerError) as exc_info:
        runner.wait(type("H", (), {"opaque": failed.id, "run_id": "rp2", "backend": "runpod"})())
    assert exc_info.value.failure.category == "execution_failed"
