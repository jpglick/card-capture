"""RemoteRuntime: handles submission to a serverless provider."""
from __future__ import annotations

from typing import Protocol

from .request import PipelineRunRequest, PipelineRunResult


class RemoteRuntime(Protocol):
    def submit(self, request: PipelineRunRequest) -> str:
        """Submit a job and return a provider-specific job_id."""
        ...

    def get_result(self, job_id: str) -> PipelineRunResult:
        """Fetch result from the provider. Blocks if necessary."""
        ...
