"""Strict guard uses monkeypatch.context() to verify forbidden ops fail.

The guard does NOT globally patch torch in production; that would break
third-party libraries. In production, StrictGpuRuntime exposes only safe
APIs. In tests, monkeypatch.context() proves the behavioral contract.
"""
from __future__ import annotations

import pytest
import torch

from card_capture.pipeline.telemetry import InMemoryTelemetry
from card_capture.runtime.guards import (
    StrictGuardActive,
    raise_forbidden_call,
    strict_section,
)


def test_strict_section_traps_tensor_cpu(monkeypatch):
    telemetry = InMemoryTelemetry()
    original_cpu = torch.Tensor.cpu
    with strict_section(telemetry=telemetry):
        with monkeypatch.context() as m:
            m.setattr(torch.Tensor, "cpu", raise_forbidden_call("torch.Tensor.cpu"))
            t = torch.zeros((1,))
            with pytest.raises(StrictGuardActive):
                t.cpu()
    # Outside the context, torch.Tensor.cpu is restored.
    assert torch.Tensor.cpu is original_cpu
    # Violation recorded.
    assert any(e.kind == "contract_violation" for e in telemetry.events)


def test_strict_section_traps_cv2_imread(monkeypatch):
    import cv2
    telemetry = InMemoryTelemetry()
    original = cv2.imread
    with strict_section(telemetry=telemetry):
        with monkeypatch.context() as m:
            m.setattr(cv2, "imread", raise_forbidden_call("cv2.imread"))
            with pytest.raises(StrictGuardActive):
                cv2.imread("does_not_matter.png")
    assert cv2.imread is original


def test_violation_carries_stable_code():
    telemetry = InMemoryTelemetry()
    with strict_section(telemetry=telemetry):
        try:
            raise_forbidden_call("torch.Tensor.numpy")()
        except StrictGuardActive:
            pass
    violations = [e for e in telemetry.events if e.kind == "contract_violation"]
    assert violations
    assert violations[0].payload["code"] == "forbidden_call:torch.Tensor.numpy"
