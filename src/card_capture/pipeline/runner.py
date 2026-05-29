"""PipelineRunner: handles local or remote submission."""
from __future__ import annotations

import dataclasses
from typing import Protocol

from .request import PipelineRunRequest, PipelineRunResult


@dataclasses.dataclass(frozen=True)
class PipelineRunHandle:
    run_id: str
    backend: str                # "local", "runpod", "beam", "vastai"
    opaque: str = ""            # provider-specific job id, opaque to callers


@dataclasses.dataclass(frozen=True)
class PipelineRunStatus:
    state: str                  # "pending", "running", "succeeded", "failed", "cancelled"
    progress: float = 0.0       # 0.0..1.0
    detail: str = ""


class PipelineRunner(Protocol):
    """Synchronous interface for local or simple remote submission."""

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle: ...
    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult: ...
    def cancel(self, handle: PipelineRunHandle) -> None: ...


class AsyncPipelineRunner(Protocol):
    """Asynchronous interface for providers with REST APIs (Beam, RunPod)."""

    async def submit_async(self, request: PipelineRunRequest) -> PipelineRunHandle: ...
    async def wait_async(self, handle: PipelineRunHandle) -> PipelineRunResult: ...
    async def cancel_async(self, handle: PipelineRunHandle) -> None: ...
