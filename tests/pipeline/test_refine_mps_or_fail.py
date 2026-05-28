# tests/pipeline/test_refine_mps_or_fail.py
import pytest
from pipeline.steps.refine import _make_kornia_normalizer


class _Boom:
    def __init__(self, *a, **k):
        raise RuntimeError("kornia init failed")


def test_gpu_warp_failure_raises_not_cpu_fallback():
    # On a GPU device, a Kornia init failure must raise (MPS-or-fail),
    # never silently downgrade to the CPU normalizer.
    with pytest.raises(RuntimeError, match="kornia"):
        _make_kornia_normalizer(_Boom, use_kornia=True, device="mps",
                                width=750, height=1050)


def test_explicit_cpu_allows_none_fallback():
    # Explicit CPU is an intentional override, so None (CPU normalizer) is allowed.
    out = _make_kornia_normalizer(_Boom, use_kornia=True, device="cpu",
                                  width=750, height=1050)
    assert out is None


def test_use_kornia_disabled_returns_none():
    out = _make_kornia_normalizer(_Boom, use_kornia=False, device="mps",
                                  width=750, height=1050)
    assert out is None
