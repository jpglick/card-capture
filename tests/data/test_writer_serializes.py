"""Writer queue serializes concurrent writes from multiple submitters."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer, Write


def _init_db(path: Path) -> None:
    conn = open_connection(path)
    conn.execute("CREATE TABLE IF NOT EXISTS counts (k TEXT PRIMARY KEY, n INTEGER)")
    conn.execute("INSERT OR REPLACE INTO counts(k, n) VALUES ('total', 0)")
    conn.close()


def test_writer_serializes_concurrent_increments(tmp_path):
    db = tmp_path / "wtest.db"
    _init_db(db)

    writer = Writer(db_path=db)
    writer.start()
    try:
        def submit_increment():
            writer.submit(Write(
                sql="UPDATE counts SET n = n + 1 WHERE k = 'total'",
                params=(),
            ))

        threads = [threading.Thread(target=submit_increment) for _ in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()
        writer.flush()
    finally:
        writer.stop()

    conn = open_connection(db, read_only=True)
    n = conn.execute("SELECT n FROM counts WHERE k='total'").fetchone()[0]
    assert n == 50, f"expected 50 increments, got {n} (write was not serialized)"


def test_writer_processes_in_submit_order(tmp_path):
    db = tmp_path / "order.db"
    conn = open_connection(db)
    conn.execute("CREATE TABLE events (i INTEGER PRIMARY KEY AUTOINCREMENT, v INTEGER)")
    conn.close()

    writer = Writer(db_path=db)
    writer.start()
    try:
        for v in range(20):
            writer.submit(Write(sql="INSERT INTO events(v) VALUES (?)", params=(v,)))
        writer.flush()
    finally:
        writer.stop()

    conn = open_connection(db, read_only=True)
    rows = conn.execute("SELECT v FROM events ORDER BY i").fetchall()
    assert [r[0] for r in rows] == list(range(20))
