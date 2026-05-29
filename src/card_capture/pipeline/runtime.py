"""PipelineRuntime: executes one run in one process."""
from __future__ import annotations

from typing import Protocol

from .request import PipelineRunRequest, PipelineRunResult


class PipelineRuntime(Protocol):
    def run(self, request: PipelineRunRequest) -> PipelineRunResult: ...
