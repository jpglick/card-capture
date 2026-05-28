"""RunPod serverless backend."""
from __future__ import annotations

import time
from typing import Any, Mapping

from card_capture.pipeline.request import PipelineRunRequest, PipelineRunResult, RunManifest
from card_capture.pipeline.remote import RemoteRuntime


class RunpodRuntime(RemoteRuntime):
    def __init__(self, api_key: str, endpoint_id: str) -> None:
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self._client = None

    def _get_client(self):
        if self._client is None:
            import runpod
            runpod.api_key = self.api_key
            self._client = runpod.Endpoint(self.endpoint_id)
        return self._client

    def submit(self, request: PipelineRunRequest) -> str:
        client = self._get_client()
        job = client.run(request.to_dict())
        return job.id

    def get_result(self, job_id: str) -> PipelineRunResult:
        client = self._get_client()
        job = client.get_job(job_id)
        
        while job.status() not in ("COMPLETED", "FAILED", "CANCELLED"):
            time.sleep(1.0)
            job = client.get_job(job_id)
            
        if job.status() == "FAILED":
            raise RuntimeError(f"RunPod job {job_id} failed: {job.error()}")
            
        output = job.output()
        manifest = RunManifest.from_json(output["manifest_json"])
        return PipelineRunResult(manifest=manifest)
