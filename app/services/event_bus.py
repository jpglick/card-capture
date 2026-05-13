"""In-process event bus for broadcasting SSE events to connected clients.

Each run gets its own set of subscriber queues.  The pipeline runner emits
events (possibly from a background thread); the SSE endpoint drains them and
forwards to the HTTP client.

Thread-safety note
------------------
``asyncio.Queue`` is not thread-safe by itself — calling ``put_nowait`` from
a thread other than the one running the event loop does not reliably wake up
the waiting coroutine.  We therefore store the event loop alongside each
subscriber queue and use ``loop.call_soon_threadsafe`` when emitting from any
calling context.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple, Optional


@dataclass
class Event:
    name: str
    payload: Optional[dict] = field(default=None)
    stage: Optional[str] = field(default=None)


class _Subscriber(NamedTuple):
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue


class EventBus:
    """Thread-safe / asyncio-safe event bus keyed by ``run_id``."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[_Subscriber]] = defaultdict(list)

    def subscribe(self, run_id: str) -> asyncio.Queue:
        """Return a new queue that will receive all events for *run_id*.

        Must be called from within a running asyncio event loop so that the
        loop reference can be captured for thread-safe delivery.
        """
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[run_id].append(_Subscriber(loop=loop, queue=q))
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        """Remove *q* from the subscriber list for *run_id*."""
        self._subscribers[run_id] = [
            s for s in self._subscribers[run_id] if s.queue is not q
        ]

    def emit(self, run_id: str, event: Event) -> None:
        """Push *event* to all subscribers of *run_id*.

        Safe to call from any thread — uses ``loop.call_soon_threadsafe`` so
        that the target event loop is properly woken up even when the caller
        is not running in that loop.  If no one is subscribed the event is
        silently dropped.
        """
        for sub in list(self._subscribers.get(run_id, [])):
            sub.loop.call_soon_threadsafe(sub.queue.put_nowait, event)
