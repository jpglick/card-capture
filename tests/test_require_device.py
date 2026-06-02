# tests/test_require_device.py
import pytest
import torch
from card_capture.core import gpu_utils


def test_require_device_mps_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS .*not available"):
        gpu_utils.require_device("mps")


def test_require_device_returns_requested_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert gpu_utils.require_device("mps").type == "mps"


def test_require_device_cpu_is_explicit_and_allowed():
    # Explicit CPU is an intentional override, so None (CPU normalizer) is allowed.
    assert gpu_utils.require_device("cpu").type == "cpu"
