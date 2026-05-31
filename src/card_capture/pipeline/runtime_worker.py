from __future__ import annotations

import queue
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class RuntimeWorker:
    """
    A narrow guarded runtime worker that executes callables on a dedicated thread.
    Used to ensure GPU-bound operations (like tracker inference) run on a specific thread.
    """

    def __init__(self) -> None:
        self._work_queue: queue.Queue[
            tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], _Result[Any]] | None
        ] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="_worker", daemon=True)
        self._running = False

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._thread.start()

    def stop(self) -> None:
        if self._running:
            self._running = False
            self._work_queue.put(None)  # Sentinel to stop
            self._thread.join()

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if not self._running:
            raise RuntimeError("RuntimeWorker is not running")
        result: _Result[T] = _Result()
        self._work_queue.put((func, args, kwargs, result))
        return result.get()

    def _run(self) -> None:
        while True:
            item = self._work_queue.get()
            if item is None:
                self._work_queue.task_done()
                break

            func, args, kwargs, result = item
            try:
                val = func(*args, **kwargs)
                result.set_value(val)
            except Exception as e:
                result.set_exception(e)
            finally:
                self._work_queue.task_done()


class _Result:
    def __init__(self) -> None:
        self._value: Any = None
        self._exception: Exception | None = None
        self._event = threading.Event()

    def set_value(self, value: Any) -> None:
        self._value = value
        self._event.set()

    def set_exception(self, exception: Exception) -> None:
        self._exception = exception
        self._event.set()

    def get(self) -> Any:
        self._event.wait()
        if self._exception:
            raise self._exception
        return self._value
