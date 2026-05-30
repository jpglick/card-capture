"""Single-writer queue for SQLite.

Spec Section 5: SQLite WAL allows concurrent readers but one writer at a
time. V5.5 routes all writes — from pipeline runtime, FastAPI handlers,
and harness — through this writer. An in-process queue serializes writes
on a dedicated worker thread; the worker holds the only write connection.

For cross-process callers (e.g., the FastAPI app in a separate uvicorn
worker), the same `Writer` API can be backed by an IPC queue in a later
iteration; the public submit/flush/stop surface is process-agnostic.
"""
from __future__ import annotations

import dataclasses
import queue
import sqlite3
import threading
from pathlib import Path

from .connection import open_connection


from contextlib import contextmanager
from typing import Iterator, Optional


@dataclasses.dataclass(frozen=True)
class Write:
    sql: str
    params: tuple = ()


_SENTINEL = object()


class WriterPoisonedError(RuntimeError):
    """Raised by submit/submit_returning once the writer has hit a write error.

    The original write error is preserved as ``__cause__`` so callers can
    inspect the root failure.
    """


class Writer:
    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        # Set once, under _lock, the first time a write raises in _loop. After
        # this the writer is "poisoned": it no longer executes writes, fails
        # queued/future returning-Futures, and discards fire-and-forget writes.
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._loop, name="card-capture-writer", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._thread is None:
                return
            self._q.put(_SENTINEL)
            self._thread.join()
            self._thread = None
            if self._error is not None:
                err, self._error = self._error, None
                raise err

    def submit(self, write: Write) -> None:
        if self._thread is None:
            raise RuntimeError("Writer.start() before submit()")
        with self._lock:
            err = self._error
        if err is not None:
            # Poisoned: fail fast rather than enqueue work that will be
            # discarded (or, worse, let the caller assume it was applied).
            raise WriterPoisonedError(
                "Writer is poisoned by a prior write error"
            ) from err
        self._q.put(write)

    def submit_returning(self, write: Write) -> "concurrent.futures.Future[int]":
        """Submit a write whose ``lastrowid`` we need.

        The internal writer thread executes the statement, calls
        ``cursor.lastrowid``, and resolves the returned Future. Use
        ``.result()`` on the call site to block until done.

        Once the writer is poisoned this returns an already-failed Future
        (its ``.result()`` re-raises the writer error) so callers never block.
        """
        import concurrent.futures
        if self._thread is None:
            raise RuntimeError("Writer.start() before submit_returning()")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            err = self._error
        if err is not None:
            # Poisoned: return an already-failed Future so .result() re-raises
            # immediately instead of blocking on a queue no one will drain.
            poison = WriterPoisonedError("Writer is poisoned by a prior write error")
            poison.__cause__ = err
            fut.set_exception(poison)
            return fut
        self._q.put(("__returning__", write, fut))
        return fut

    def flush(self) -> None:
        """Block until the queue is empty (best-effort)."""
        self._q.join()

    @contextmanager
    def serialize(self) -> Iterator[None]:
        """Acquire the writer lock for callers that must perform a direct
        synchronous write (e.g., to read back an autoincrement id).

        While the lock is held the worker thread keeps draining the queue,
        so this only protects against concurrent direct writers, not against
        the worker. Callers MUST close any connection they opened before
        releasing the lock.
        """
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def _loop(self) -> None:
        conn = open_connection(self._db_path)
        try:
            while True:
                item = self._q.get()
                try:
                    if item is _SENTINEL:
                        return

                    if self._error is not None:
                        # Poisoned: never execute another write. Drain the
                        # queue so no caller blocks forever — fail returning
                        # Futures with the recorded error, discard fire-and-
                        # forget writes. The loop keeps running until stop()
                        # enqueues the sentinel.
                        if isinstance(item, tuple) and item and item[0] == "__returning__":
                            _, _write, fut = item
                            if not fut.done():
                                fut.set_exception(self._error)
                        continue

                    if isinstance(item, tuple) and item and item[0] == "__returning__":
                        _, write, fut = item
                        try:
                            cur = conn.execute(write.sql, write.params)
                            fut.set_result(cur.lastrowid)
                        except BaseException as exc:
                            # A returning write that fails resolves only its own
                            # Future; the writer is NOT poisoned by it.
                            fut.set_exception(exc)
                        continue

                    write: Write = item
                    conn.execute(write.sql, write.params)
                except BaseException as exc:  # noqa: BLE001
                    # First fire-and-forget write error: record it (preserving
                    # the FIRST error for stop() to re-raise) and transition to
                    # drain-and-fail mode instead of dying with the queue full.
                    with self._lock:
                        if self._error is None:
                            self._error = exc
                finally:
                    self._q.task_done()
        finally:
            conn.close()
