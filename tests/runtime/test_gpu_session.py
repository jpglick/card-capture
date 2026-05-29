from __future__ import annotations

import pytest
import torch

from card_capture.pipeline.telemetry import NoopTelemetry
from card_capture.runtime.gpu_session import GpuSession, MissingGpuError


def test_session_requires_device():
    with pytest.raises(TypeError):
        GpuSession()  # type: ignore[call-arg]


def test_session_records_capability():
    sess = GpuSession(device=torch.device("cpu"), strict=False, telemetry=NoopTelemetry())
    assert sess.device.type == "cpu"
    assert sess.strict is False


def test_strict_session_rejects_cpu_device():
    with pytest.raises(MissingGpuError):
        GpuSession(device=torch.device("cpu"), strict=True, telemetry=NoopTelemetry())
