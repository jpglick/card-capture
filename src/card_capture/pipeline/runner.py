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
    """Synchronous interface. Async wrapper added in Phase 5 for remote adapters."""

    def submit(self, request: PipelineRunRequest) -> PipelineRunHandle: ...
    def wait(self, handle: PipelineRunHandle) -> PipelineRunResult: ...
    def cancel(self, handle: PipelineRunHandle) -> None: ...
