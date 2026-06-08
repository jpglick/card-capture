"""OpenTelemetry SDK bootstrap.

``OpenTelemetryAdapter`` only records into whatever global Tracer/Meter provider
is configured. Without a provider + exporter the OTel API hands back no-op
instruments and everything is silently dropped — so a run's spans/metrics go
nowhere unless something calls :func:`configure_telemetry` first.

Exporters are opt-in and selected from the environment so the default is quiet
(no console spam, no hard dependency on a running collector):

* ``OTEL_EXPORTER_OTLP_ENDPOINT`` set  -> export traces+metrics over OTLP/HTTP
  (requires the ``opentelemetry-exporter-otlp-proto-http`` package).
* ``CARD_CAPTURE_OTEL_CONSOLE`` truthy -> export to stdout (local debugging).

Call once at process startup. Durable per-run capture is handled separately by
``DbTelemetry`` writing to ``telemetry_events``; OTel is for live trace/metric
backends.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

_SERVICE_NAME = "card-capture"

# Module-level singleton state for the production (environ=None) call path.
_configured = False
_enabled = False

_FALSEY = {"", "0", "false", "no", "off"}


def _truthy(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() not in _FALSEY


@dataclass(frozen=True)
class ExporterPlan:
    """What exporters the environment asks for."""

    enabled: bool
    otlp_endpoint: Optional[str]
    console: bool


def resolve_exporter_plan(environ: Mapping[str, str]) -> ExporterPlan:
    """Decide which exporters to install from environment variables (pure)."""
    endpoint = (environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip() or None
    console = _truthy(environ.get("CARD_CAPTURE_OTEL_CONSOLE"))
    return ExporterPlan(enabled=bool(endpoint) or console, otlp_endpoint=endpoint, console=console)


def configure_telemetry(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Configure global OTel providers + exporters. Returns whether OTel is live.

    With ``environ=None`` (production) the result is cached so repeated startup
    calls are cheap and providers are set only once. Passing an explicit mapping
    (tests) bypasses the cache and always re-evaluates.
    """
    global _configured, _enabled
    explicit = environ is not None
    if _configured and not explicit:
        return _enabled

    env = os.environ if environ is None else environ
    plan = resolve_exporter_plan(env)

    enabled = False
    if plan.enabled:
        try:
            enabled = _install_providers(plan)
        except Exception:  # never let telemetry setup take down the app
            logger.warning(
                "OpenTelemetry setup failed; continuing without OTel export", exc_info=True
            )
            enabled = False

    if not explicit:
        _configured = True
        _enabled = enabled
    return enabled


def _install_providers(plan: ExporterPlan) -> bool:
    """Install span processors / metric readers per ``plan``. Returns installed."""
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    span_processors = []
    metric_readers = []

    if plan.otlp_endpoint:
        try:
            # The exporters read OTEL_EXPORTER_OTLP_ENDPOINT (and friends) from
            # the environment themselves; no need to pass the endpoint through.
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            span_processors.append(BatchSpanProcessor(OTLPSpanExporter()))
            metric_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
        except Exception:
            logger.warning(
                "OTEL_EXPORTER_OTLP_ENDPOINT is set but the OTLP exporter package "
                "is unavailable; install 'opentelemetry-exporter-otlp-proto-http'",
                exc_info=True,
            )

    if plan.console:
        span_processors.append(SimpleSpanProcessor(ConsoleSpanExporter()))
        metric_readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))

    if not span_processors and not metric_readers:
        return False

    resource = Resource.create({"service.name": _SERVICE_NAME})

    # Reuse an existing SDK TracerProvider if one is already global (set_* only
    # takes effect once); otherwise create and install ours.
    current_tp = trace.get_tracer_provider()
    if isinstance(current_tp, TracerProvider):
        tracer_provider = current_tp
    else:
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
    for processor in span_processors:
        tracer_provider.add_span_processor(processor)

    # MeterProvider readers can only be supplied at construction. Only install
    # ours if the global provider isn't already an SDK one.
    if not isinstance(metrics.get_meter_provider(), MeterProvider):
        metrics.set_meter_provider(
            MeterProvider(resource=resource, metric_readers=metric_readers)
        )

    return True
