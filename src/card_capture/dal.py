from __future__ import annotations

import base64
import json
import sqlite3
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .models import CardDetection, CornerDetection, PerformanceTelemetry, QualityScore


class DataAccessLayer(Protocol):
    """Protocol for data access operations."""

    def add_video(self, source_path: str, file_hash: str, duration_ms: int, width: int, height: int, status: str = "processing") -> int:
        ...

    def add_pipeline_event(self, video_id: int, frame_index: int, timestamp_ms: int, event_type: str, data: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None, stage_id: Optional[str] = None, artifact_ref: Optional[str] = None) -> None:
        ...

    # TODO: Add other methods as needed


class SQLiteDAL:
    """Standard SQLite implementation of DAL."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def add_video(self, source_path: str, file_hash: str, duration_ms: int, width: int, height: int, status: str = "processing") -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO videos (source_path, file_hash, duration_ms, width, height, status) VALUES (?, ?, ?, ?, ?, ?)",
                (source_path, file_hash, duration_ms, width, height, status),
            )
            return int(cursor.lastrowid)

    def add_pipeline_event(self, video_id: int, frame_index: int, timestamp_ms: int, event_type: str, data: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None, stage_id: Optional[str] = None, artifact_ref: Optional[str] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_events (video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, data_json, artifact_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, json.dumps(data) if data else None, artifact_ref),
            )


class SingleWriterDAL:
    """Thread-safe DAL that uses a dedicated background thread for all writes.
    
    This prevents 'database is locked' errors in multi-threaded environments (like UnifiedRuntime).
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._write_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._write_worker, name="dal-writer", daemon=True)
        self._worker_thread.start()

    def _write_worker(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL") # Better concurrency
        
        while not (self._stop_event.is_set() and self._write_queue.empty()):
            try:
                task = self._write_queue.get(timeout=0.1)
                sql, params, result_queue = task
                try:
                    cursor = conn.execute(sql, params)
                    conn.commit()
                    if result_queue:
                        result_queue.put(cursor.lastrowid)
                except Exception as e:
                    if result_queue:
                        result_queue.put(e)
                finally:
                    self._write_queue.task_done()
            except queue.Empty:
                continue
        conn.close()

    def _execute_write(self, sql: str, params: tuple, wait: bool = False) -> Any:
        result_queue = queue.Queue() if wait else None
        self._write_queue.put((sql, params, result_queue))
        if wait:
            result = result_queue.get()
            if isinstance(result, Exception):
                raise result
            return result
        return None

    def add_video(self, source_path: str, file_hash: str, duration_ms: int, width: int, height: int, status: str = "processing") -> int:
        return self._execute_write(
            "INSERT INTO videos (source_path, file_hash, duration_ms, width, height, status) VALUES (?, ?, ?, ?, ?, ?)",
            (source_path, file_hash, duration_ms, width, height, status),
            wait=True
        )

    def add_pipeline_event(self, video_id: int, frame_index: int, timestamp_ms: int, event_type: str, data: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None, stage_id: Optional[str] = None, artifact_ref: Optional[str] = None) -> None:
        self._execute_write(
            """
            INSERT INTO pipeline_events (video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, data_json, artifact_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, json.dumps(data) if data else None, artifact_ref),
            wait=False # Events are fire-and-forget
        )

    def shutdown(self):
        self._stop_event.set()
        self._worker_thread.join()
