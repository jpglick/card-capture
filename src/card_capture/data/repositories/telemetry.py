"""telemetry_events repository (durable mirror of in-memory telemetry)."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class TelemetryRepository:
    def __init__(self, writer: Writer, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def record_event(self, run_id: str | None, kind: str, payload: Mapping[str, object]) -> None:
        self._writer.submit(Write(
            sql="INSERT INTO telemetry_events(run_id, kind, payload, at_ms) VALUES (?, ?, ?, ?)",
            params=(run_id, kind, json.dumps(dict(payload)), int(time.time() * 1000)),
        ))

    def list_for_run(self, run_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT run_id, kind, payload, at_ms FROM telemetry_events WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [{"run_id": r[0], "kind": r[1], "payload": r[2], "at_ms": r[3]} for r in rows]
