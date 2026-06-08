"""PipelineRunner assembles the right telemetry sinks from its configuration."""
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from app.services.pipeline_runner import PipelineRunner
from app.services.pipeline_telemetry import DbTelemetry
from card_capture.pipeline.telemetry import (
    CompositeTelemetry,
    InMemoryTelemetry,
    OpenTelemetryAdapter,
)


def _runner(**kwargs) -> PipelineRunner:
    return PipelineRunner(bus=None, **kwargs)


def test_assemble_default_has_only_base_sink():
    runner = _runner()
    base = InMemoryTelemetry()
    telemetry, adapter = runner._assemble_telemetry("run_1", base)
    assert adapter is None
    assert isinstance(telemetry, CompositeTelemetry)
    assert telemetry._sinks == [base]


def test_assemble_with_otel_enabled_adds_adapter_and_returns_it_for_shutdown():
    runner = _runner(otel_enabled=True)
    base = InMemoryTelemetry()
    telemetry, adapter = runner._assemble_telemetry("run_1", base)
    assert isinstance(adapter, OpenTelemetryAdapter)
    assert adapter in telemetry._sinks


def test_assemble_with_db_repo_adds_db_sink():
    runner = _runner(telemetry_repo=object())  # truthy; DbTelemetry only stores it
    base = InMemoryTelemetry()
    telemetry, adapter = runner._assemble_telemetry("run_1", base)
    assert adapter is None
    assert any(isinstance(s, DbTelemetry) for s in telemetry._sinks)


def test_build_runner_forwards_otel_enabled():
    from types import SimpleNamespace

    from app.api.videos import _build_runner

    state = SimpleNamespace(
        event_bus=object(), db_path=None, telemetry_repo=None, otel_enabled=True
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    runner = _build_runner(request)
    assert runner.otel_enabled is True


def test_assemble_otel_import_failure_degrades_gracefully(monkeypatch):
    # If constructing the OTel adapter blows up, the run still gets telemetry.
    import card_capture.pipeline.telemetry as tel

    def _boom(*args, **kwargs):
        raise RuntimeError("otel exploded")

    monkeypatch.setattr(tel, "OpenTelemetryAdapter", _boom)
    runner = _runner(otel_enabled=True)
    base = InMemoryTelemetry()
    telemetry, adapter = runner._assemble_telemetry("run_1", base)
    assert adapter is None
    assert telemetry._sinks == [base]
