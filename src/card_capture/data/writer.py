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


@dataclasses.dataclass(frozen=True)
class Write:
    sql: str
    params: tuple = ()


_SENTINEL = object()


class Writer:
    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
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
        self._q.put(write)

    def flush(self) -> None:
        """Block until the queue is empty (best-effort)."""
        self._q.join()

    def _loop(self) -> None:
        conn = open_connection(self._db_path)
        try:
            while True:
                item = self._q.get()
                try:
                    if item is _SENTINEL:
                        return
                    write: Write = item
                    conn.execute(write.sql, write.params)
                except BaseException as exc:  # noqa: BLE001
                    self._error = exc
                    return
                finally:
                    self._q.task_done()
        finally:
            conn.close()
