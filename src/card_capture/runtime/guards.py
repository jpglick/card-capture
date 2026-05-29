"""Strict-mode runtime guard.

The guard is **not** a global monkeypatch in production. It records contract
violations into telemetry when invoked. Tests use `monkeypatch.context()` to
prove that, when a forbidden call IS made, the guard fires and records the
violation with a stable code.

Inside a `strict_section`, callers may install patched versions of forbidden
operations using `raise_forbidden_call(name)` as the replacement; the patched
callable raises `StrictGuardActive` and records the violation.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Callable, Iterator

from card_capture.pipeline.telemetry import PipelineTelemetry


_local = threading.local()


class StrictGuardActive(RuntimeError):
    """Raised when a forbidden call fires inside a strict section."""


@contextlib.contextmanager
def strict_section(telemetry: PipelineTelemetry) -> Iterator[None]:
    """Mark the current thread as inside a strict GPU section."""
    prev = getattr(_local, "telemetry", None)
    _local.telemetry = telemetry
    try:
        yield
    finally:
        _local.telemetry = prev


def _current_telemetry() -> PipelineTelemetry | None:
    return getattr(_local, "telemetry", None)


def raise_forbidden_call(name: str) -> Callable[..., object]:
    """Return a callable that records a violation and raises StrictGuardActive.

    Intended for use with `monkeypatch.context()` in tests:

        with monkeypatch.context() as m:
            m.setattr(torch.Tensor, "cpu", raise_forbidden_call("torch.Tensor.cpu"))
            ...
    """
    def _trap(*args, **kwargs):
        sink = _current_telemetry()
        if sink is not None:
            sink.contract_violation(f"forbidden_call:{name}", {"name": name})
        raise StrictGuardActive(f"Forbidden call inside strict_section: {name}")

    return _trap
