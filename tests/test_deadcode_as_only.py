import pytest
import torch
import inspect

from card_capture.core import gpu_utils
from card_capture import fuser, scoring
from card_capture.core.config import PipelineConfig


def test_config_has_allow_cpu_fallback_default_false():
    assert PipelineConfig().allow_cpu_fallback is False


def test_get_device_returns_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    # The function will likely take the arg now
    assert gpu_utils.get_device(allow_cpu_fallback=False).type == "mps"


def test_get_device_hard_fails_without_mps_and_no_flag(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        gpu_utils.get_device(allow_cpu_fallback=False)


def test_get_device_allows_cpu_with_flag(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert gpu_utils.get_device(allow_cpu_fallback=True).type == "cpu"


def test_fuser_has_no_cuda_branch():
    src = inspect.getsource(fuser)
    # We ignore comments in this check if possible, but the spec says "not in src.lower()"
    assert "cuda" not in src.lower()
    assert "CC_CUDA_ALLOW_CPU_FALLBACK" not in src


def test_scoring_has_no_cuda_branch():
    src = inspect.getsource(scoring)
    assert "cuda" not in src.lower()
