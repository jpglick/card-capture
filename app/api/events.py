"""SSE channel — `/events/{run_id}`.

Streams events from the in-process ``EventBus`` to the HTTP client using
``sse-starlette``.  The stream terminates automatically when a
``run_completed`` or ``run_failed`` event is emitted or when the client
disconnects.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.services.event_bus import Event, EventBus

router = APIRouter()


async def _stream(request: Request, bus: EventBus, run_id: str):
    q = bus.subscribe(run_id)
    try:
        while True:
            if await request.is_disconnected():
                break
            event: Event = await q.get()
            yield {
                "event": event.name,
                "data": json.dumps({"stage": event.stage, "payload": event.payload}),
            }
            if event.name in {"run_completed", "run_failed"}:
                break
    finally:
        bus.unsubscribe(run_id, q)


@router.get("/{run_id}")
async def events(request: Request, run_id: str):
    """Subscribe to real-time SSE progress for *run_id*."""
    bus: EventBus = request.app.state.event_bus
    return EventSourceResponse(_stream(request, bus, run_id))
