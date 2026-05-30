"""Background frame producer: decode on a worker thread, consume on the main thread.

The ``sample`` stage starts a :class:`FrameProducer`, which runs a sampler's
``.sample()`` (CPU/ffmpeg decode) on a daemon thread and pushes each
``FrameSample`` onto a bounded queue. The ``detect`` stage then drains the queue
and runs inference, so decode overlaps inference instead of completing first.
"""
from __future__ import annotations

import queue
import threading
from typing import Iterator, Optional, Protocol

from card_capture.models import FrameSample


class _Samplerish(Protocol):
    def sample(self) -> Iterator[FrameSample]:
        ...


_SENTINEL = object()


class FrameProducer:
    """Run ``sampler.sample()`` on a daemon thread, feeding a bounded queue."""

    def __init__(self, sampler: _Samplerish, *, maxsize: int = 32) -> None:
        self._sampler = sampler
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._first_ready = threading.Event()

    @property
    def queue(self) -> "queue.Queue":
        return self._queue

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def start(self) -> "FrameProducer":
        self._thread = threading.Thread(
            target=self._run, name="frame-producer", daemon=True
        )
        self._thread.start()
        return self

    def wait_first(self, timeout: Optional[float] = None) -> bool:
        """Block until the first frame is enqueued (or the run ended)."""
        return self._first_ready.wait(timeout)

    def _run(self) -> None:
        try:
            for frame in self._sampler.sample():
                if self._stop.is_set():
                    break
                self._queue.put(frame)
                self._first_ready.set()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer
            self._error = exc
        finally:
            self._first_ready.set()
            self._queue.put(_SENTINEL)

    def __iter__(self) -> Iterator[FrameSample]:
        """Yield frames until the sentinel, then join and re-raise producer errors."""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            yield item
        self.join()
        if self._error is not None:
            raise self._error

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the producer to stop, draining so a blocked ``put`` unblocks."""
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        for _ in range(max(1, int(timeout / 0.05))):
            if not thread.is_alive():
                break
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            thread.join(timeout=0.05)

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
