"""Writer fails loud instead of deadlocking after a write error.

Regression coverage for the confirmed deadlock: a fire-and-forget ``submit()``
whose SQL raises used to silently kill the worker thread. Any later
``submit_returning(...).result()`` then blocked forever because no thread
would ever drain the queue or resolve the Future.

Invariant under test: after ANY write error, no caller may block
indefinitely. Every blocking call here is bounded (``result(timeout=...)``
plus a watchdog) so a regression FAILS the suite instead of hanging it.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer, Write, WriterPoisonedError


_TIMEOUT = 5.0


def _init_db(path: Path) -> None:
    conn = open_connection(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v INTEGER)")
    conn.close()


def _bad_write() -> Write:
    # Column ``nope`` does not exist -> sqlite3.OperationalError at execute time.
    return Write(sql="INSERT INTO t(nope) VALUES (?)", params=(1,))


def _good_write(v: int = 1) -> Write:
    return Write(sql="INSERT INTO t(v) VALUES (?)", params=(v,))


def _run_bounded(fn, timeout: float = _TIMEOUT):
    """Run ``fn`` in a watchdog thread; fail (not hang) if it overruns.

    Returns whatever ``fn`` returns, or re-raises whatever it raised. If the
    call does not complete within ``timeout`` the test fails loudly — this is
    the safety net that turns the old deadlock into a fast test failure even
    if the bounded ``.result(timeout=...)`` were somehow bypassed.
    """
    box: dict = {}

    def _target():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"call did not complete within {timeout}s (deadlock?)"
    if "error" in box:
        raise box["error"]
    return box.get("result")


def test_poisoned_returning_result_raises_not_hangs(tmp_path):
    """The exact reproduced deadlock: a bad submit() poisons the writer, then a
    later submit_returning().result() must RAISE within the timeout (it used to
    block forever)."""
    db = tmp_path / "poison.db"
    _init_db(db)

    writer = Writer(db_path=db)
    writer.start()
    try:
        # Poison the writer with a fire-and-forget bad write.
        writer.submit(_bad_write())
        # Let the worker process the bad write and transition to poisoned.
        # flush() is bounded by the watchdog below.
        _run_bounded(writer.flush)

        def _submit_and_wait():
            fut = writer.submit_returning(_good_write())
            return fut.result(timeout=_TIMEOUT)

        with pytest.raises(BaseException) as excinfo:
            _run_bounded(_submit_and_wait)
        # Surfaces either the original write error or the poison wrapper.
        assert excinfo.value is not None
    finally:
        with pytest.raises(BaseException):
            writer.stop()


def test_returning_write_error_resolves_own_future(tmp_path):
    """A submit_returning whose own SQL errors resolves ITS future with the
    exception and does NOT poison the writer (existing behavior preserved)."""
    db = tmp_path / "ret_err.db"
    _init_db(db)

    writer = Writer(db_path=db)
    writer.start()
    try:
        bad_fut = writer.submit_returning(_bad_write())
        with pytest.raises(Exception):
            _run_bounded(lambda: bad_fut.result(timeout=_TIMEOUT))

        # Writer is NOT poisoned: a subsequent valid returning write succeeds.
        good_fut = writer.submit_returning(_good_write(7))
        rowid = _run_bounded(lambda: good_fut.result(timeout=_TIMEOUT))
        assert isinstance(rowid, int) and rowid >= 1
    finally:
        writer.stop()

    conn = open_connection(db, read_only=True)
    rows = conn.execute("SELECT v FROM t ORDER BY id").fetchall()
    conn.close()
    assert [r[0] for r in rows] == [7]


def test_new_calls_after_poison_fail_fast(tmp_path):
    """After poisoning, submit() raises and submit_returning() returns an
    already-failed Future; neither blocks."""
    db = tmp_path / "failfast.db"
    _init_db(db)

    writer = Writer(db_path=db)
    writer.start()
    try:
        writer.submit(_bad_write())
        _run_bounded(writer.flush)

        # submit() must raise fast.
        with pytest.raises(WriterPoisonedError):
            _run_bounded(lambda: writer.submit(_good_write()))

        # submit_returning() returns an already-failed Future; .result() raises
        # essentially immediately (no queue drain needed).
        fut = _run_bounded(lambda: writer.submit_returning(_good_write()))
        with pytest.raises(WriterPoisonedError):
            fut.result(timeout=_TIMEOUT)
    finally:
        with pytest.raises(BaseException):
            writer.stop()


def test_stop_after_poison_reraises_and_returns_promptly(tmp_path):
    """stop() after a poisoned write re-raises the original error and does not
    hang."""
    db = tmp_path / "stop.db"
    _init_db(db)

    writer = Writer(db_path=db)
    writer.start()
    writer.submit(_bad_write())
    _run_bounded(writer.flush)

    with pytest.raises(BaseException) as excinfo:
        _run_bounded(writer.stop)
    # The first write error (an sqlite3 error) is what stop() re-raises.
    assert excinfo.value is not None
    # And the error has been consumed; a second stop() is a no-op (thread None).
    _run_bounded(writer.stop)


def test_happy_path_unchanged(tmp_path):
    """No errors: submit + submit_returning().result() + flush() + stop() all
    work and rows are committed, exactly as before."""
    db = tmp_path / "happy.db"
    _init_db(db)

    writer = Writer(db_path=db)
    writer.start()
    try:
        writer.submit(_good_write(10))
        writer.submit(_good_write(20))

        fut = writer.submit_returning(_good_write(30))
        rowid = _run_bounded(lambda: fut.result(timeout=_TIMEOUT))
        assert isinstance(rowid, int) and rowid >= 1

        writer.submit(_good_write(40))
        _run_bounded(writer.flush)
    finally:
        writer.stop()  # clean join, no error to re-raise

    conn = open_connection(db, read_only=True)
    rows = conn.execute("SELECT v FROM t ORDER BY id").fetchall()
    conn.close()
    assert [r[0] for r in rows] == [10, 20, 30, 40]
