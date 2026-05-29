"""RunPod serverless backend implementing PipelineRunner."""
from __future__ import annotations

import time

from card_capture.pipeline.request import PipelineRunRequest, PipelineRunResult, RunManifest
from card_capture.pipeline.runner import PipelineRunHandle, PipelineRunner
from card_capture.platforms.failures import ProviderFailure, map_provider_failure


class RunpodRunnerError(RuntimeError):
    def __init__(self, message: str, failure: ProviderFailure) -> None:
        super().__init__(message)
        self.failure = failure


class RunpodRunner(PipelineRunner):
    def __init__(self, api_key: str, endpoint_id: str, poll_interval: float = 1.0) -> None:
        import runpod

        runpod.api_key = api_key
        self._endpoint = runpod.Endpoint(endpoint_id)
        self._poll_interval = poll_interval

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        try:
            job = self._endpoint.run(request.to_dict())
        except Exception as exc:  # noqa: BLE001
            failure = map_provider_failure(provider="runpod", raw=str(exc))
            raise RunpodRunnerError(f"RunPod submit failed: {exc}", failure) from exc
        return PipelineRunHandle(run_id=request.run_id, backend="runpod", opaque=job.id)

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        job = self._endpoint.get_job(handle.opaque)
        while True:
            state = job.status()
            if state == "COMPLETED":
                out = job.output()
                manifest = RunManifest.from_json(out["manifest_json"])
                return PipelineRunResult(manifest=manifest)
            if state in ("FAILED", "CANCELLED"):
                raw = ""
                try:
                    raw = job.error() or state
                except Exception:  # noqa: BLE001
                    raw = state
                failure = map_provider_failure(provider="runpod", raw=raw)
                raise RunpodRunnerError(f"RunPod job {handle.opaque} failed: {raw}", failure)
            time.sleep(self._poll_interval)
            job = self._endpoint.get_job(handle.opaque)

    def cancel(self, handle: PipelineRunHandle) -> None:
        job = self._endpoint.get_job(handle.opaque)
        job.cancel()
