"""GpuSession: capability object required to enter GPU hot-path code."""
from __future__ import annotations

import dataclasses

import torch

from card_capture.pipeline.telemetry import PipelineTelemetry


class MissingGpuError(RuntimeError):
    """Raised when StrictGpuRuntime is constructed without a GPU device."""


@dataclasses.dataclass(frozen=True)
class GpuSession:
    device: torch.device
    strict: bool
    telemetry: PipelineTelemetry

    def __post_init__(self) -> None:
        if self.strict and self.device.type == "cpu":
            raise MissingGpuError(
                f"strict GpuSession requires a GPU device, got {self.device}"
            )
