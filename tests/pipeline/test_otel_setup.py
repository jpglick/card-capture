"""configure_telemetry() selects exporters from the environment and is safe to
call when nothing is configured."""
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from card_capture.pipeline.otel_setup import (
    configure_telemetry,
    resolve_exporter_plan,
)


def test_plan_disabled_when_nothing_configured():
    plan = resolve_exporter_plan({})
    assert plan.enabled is False
    assert plan.otlp_endpoint is None
    assert plan.console is False


def test_plan_enables_otlp_from_endpoint_env():
    plan = resolve_exporter_plan({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318"})
    assert plan.enabled is True
    assert plan.otlp_endpoint == "http://collector:4318"


def test_plan_enables_console_from_flag():
    plan = resolve_exporter_plan({"CARD_CAPTURE_OTEL_CONSOLE": "1"})
    assert plan.enabled is True
    assert plan.console is True


def test_plan_blank_endpoint_is_not_enabled():
    plan = resolve_exporter_plan({"OTEL_EXPORTER_OTLP_ENDPOINT": "   "})
    assert plan.enabled is False
    assert plan.otlp_endpoint is None


def test_configure_telemetry_disabled_returns_false():
    # No exporter env → returns False and does not raise, even if called twice.
    assert configure_telemetry(environ={}) is False
    assert configure_telemetry(environ={}) is False


def test_configure_telemetry_console_returns_true():
    from opentelemetry import metrics

    assert configure_telemetry(environ={"CARD_CAPTURE_OTEL_CONSOLE": "1"}) is True

    # configure_telemetry installs a global MeterProvider with a periodic console
    # reader (a daemon export thread). Shut it down so it does not flush to a
    # closed stdout once pytest tears down output capture.
    shutdown = getattr(metrics.get_meter_provider(), "shutdown", None)
    if callable(shutdown):
        shutdown()
