"""Local synchronous runner."""
from __future__ import annotations

from card_capture.pipeline.request import PipelineRunRequest, PipelineRunResult
from card_capture.pipeline.runner import PipelineRunHandle, PipelineRunner
from card_capture.pipeline.runtime_local import LocalPipelineRuntime


class LocalRunner(PipelineRunner):
    def __init__(self, runtime: LocalPipelineRuntime | None = None) -> None:
        self._runtime = runtime or LocalPipelineRuntime()

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle:
        return self._runtime.submit(request)

    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult:
        if handle.backend != "local":
            raise KeyError(f"unknown backend for LocalRunner: {handle.backend!r}")
        return self._runtime.wait(handle)

    def cancel(self, handle: PipelineRunHandle) -> None:
        self._runtime.cancel(handle)
