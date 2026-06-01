from __future__ import annotations

import threading

from card_capture.pipeline.runtime_worker import RuntimeWorker


def test_runtime_worker_executes_callable_on_guarded_worker_thread():
    caller = threading.get_ident()
    worker = RuntimeWorker()
    worker.start()
    try:
        worker_ident = worker.call(threading.get_ident)
    finally:
        worker.stop()
    assert worker_ident != caller


def test_runtime_worker_propagates_exceptions():
    worker = RuntimeWorker()
    worker.start()

    def failing_func():
        raise ValueError("test exception")

    try:
        import pytest
        with pytest.raises(ValueError, match="test exception"):
            worker.call(failing_func)
    finally:
        worker.stop()
