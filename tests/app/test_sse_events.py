"""SSE channel tests — event ordering and payload schema.

The ``EventBus`` is injected into the app state so we can emit events from
the test and observe them through the SSE endpoint.

TestClient limitation
---------------------
Starlette's ``TestClient`` (backed by httpx) does NOT support true streaming:
it runs the entire ASGI app to completion and buffers the full response before
returning.  For SSE this means:

  * The SSE generator must close itself (break after a terminal event) before
    ``client.stream().__enter__()`` returns.
  * Events emitted *after* the HTTP request starts are never seen unless the
    generator is already blocked waiting for them.

We work around this by using a ``ReplayEventBus`` — a test-only subclass that
pre-buffers emitted events.  When the SSE generator calls ``bus.subscribe()``,
it receives all buffered events immediately (via ``asyncio.Queue.put_nowait``),
so the generator can process them, hit the terminal event, break, and let the
ASGI app return.  ``iter_lines()`` then reads from the already-complete buffer.
"""
from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.event_bus import Event, EventBus


# ---------------------------------------------------------------------------
# Test double: replays buffered events to late subscribers
# ---------------------------------------------------------------------------

class ReplayEventBus(EventBus):
    """Buffers every emitted event and replays them to new subscribers.

    This lets tests call ``emit()`` *before* the SSE handler subscribes, which
    is required when using the synchronous ``TestClient``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._buffer: dict[str, list[Event]] = {}

    def emit(self, run_id: str, event: Event) -> None:
        self._buffer.setdefault(run_id, []).append(event)
        # Deliver to any already-subscribed clients (standard EventBus behaviour)
        super().emit(run_id, event)

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q = super().subscribe(run_id)
        # Replay buffered events so the generator gets them immediately
        for event in self._buffer.get(run_id, []):
            q.put_nowait(event)
        return q


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(bus: EventBus) -> TestClient:
    """Return a TestClient whose app uses *bus* as its event bus."""
    app = create_app()
    app.state.event_bus = bus
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sse_emits_stage_progress_in_order():
    """Events must arrive in the order they were emitted, terminal last."""
    bus = ReplayEventBus()
    bus.emit("run_42", Event(name="stage_started", stage="detect"))
    bus.emit("run_42", Event(name="stage_completed", stage="detect"))
    bus.emit("run_42", Event(name="run_completed"))

    client = _make_client(bus)

    seen: list[str] = []
    with client.stream("GET", "/events/run_42") as response:
        for line in response.iter_lines():
            seen.append(line)
            if "run_completed" in line:
                break

    names = [line for line in seen if line.startswith("event:")]
    assert names == [
        "event: stage_started",
        "event: stage_completed",
        "event: run_completed",
    ]


def test_sse_data_lines_contain_stage_field():
    """``data:`` lines must include a JSON object with a ``stage`` key."""
    bus = ReplayEventBus()
    bus.emit("run_43", Event(name="stage_started", stage="novelty", payload={"foo": 1}))
    bus.emit("run_43", Event(name="run_completed"))

    client = _make_client(bus)

    data_lines: list[str] = []
    with client.stream("GET", "/events/run_43") as response:
        for line in response.iter_lines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
            if "run_completed" in line:
                break

    assert data_lines, "no data lines received"
    first = json.loads(data_lines[0])
    assert "stage" in first
    assert first["stage"] == "novelty"


def test_sse_terminates_on_run_failed():
    """The stream must close after a ``run_failed`` terminal event."""
    bus = ReplayEventBus()
    bus.emit("run_err", Event(name="run_failed", payload={"error": "boom"}))

    client = _make_client(bus)

    seen_events: list[str] = []
    with client.stream("GET", "/events/run_err") as response:
        for line in response.iter_lines():
            if line.startswith("event:"):
                seen_events.append(line)
            if "run_failed" in line:
                break

    assert "event: run_failed" in seen_events
