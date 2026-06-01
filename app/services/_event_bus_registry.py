"""Thread-safe registry that maps run_id -> EventBus.

The pipeline runtime imports this module to retrieve the EventBus for the
currently executing run and emit progress events back to the HTTP layer.
They look up the run_id from the ``EVENT_BUS_RUN_ID`` environment variable
that ``PipelineRunner`` sets before launching the flow.
"""
from __future__ import annotations

from threading import Lock

from app.services.event_bus import EventBus

_lock = Lock()
_buses: dict[str, EventBus] = {}


def set(run_id: str, bus: EventBus) -> None:  # noqa: A001 — intentional shadow
    """Register *bus* for *run_id*."""
    with _lock:
        _buses[run_id] = bus


def get(run_id: str) -> EventBus | None:
    """Return the bus for *run_id*, or ``None`` if not registered."""
    with _lock:
        return _buses.get(run_id)


def clear(run_id: str) -> None:
    """Deregister the bus for *run_id* (call after the run finishes)."""
    with _lock:
        _buses.pop(run_id, None)
