"""PipelineRunner: uniform submit/wait/cancel surface for local and remote backends.

Concrete implementations live in `card_capture.platforms.*`. The handle is
opaque to callers; each backend stores its provider-specific job_id in
`PipelineRunHandle.opaque`. This module replaces the earlier `RemoteRuntime`
Protocol (collapsed in Phase D); legacy callers should migrate to
`PipelineRunner`.
"""
from __future__ import annotations

import dataclasses
from typing import Protocol

from .request import PipelineRunRequest, PipelineRunResult


@dataclasses.dataclass(frozen=True)
class PipelineRunHandle:
    run_id: str
    backend: str                # "local", "runpod", "beam"
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
