"""SQLite connection management.

Connections use WAL mode and a 5-second busy_timeout. Writers must route
through card_capture.data.writer (Task 4.2) to ensure serialization;
direct write usage is allowed only inside this package for the writer's
worker thread.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def open_connection(db_path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode={'ro' if read_only else 'rwc'}"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    if not read_only:
        import time
        # Small retry loop for WAL mode setup which can race with other connections
        for _ in range(5):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc):
                    time.sleep(0.05)
                    continue
                raise
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def read_connection(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    conn = open_connection(db_path, read_only=True)
    try:
        yield conn
    finally:
        conn.close()
