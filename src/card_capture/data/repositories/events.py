"""pipeline_events repository — production schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class EventsRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def record(
        self,
        *,
        run_id: str | None,
        video_id: int | None,
        stage_id: str,
        frame_index: int,
        timestamp_ms: int,
        event_type: str,
        data: Mapping[str, object] | None = None,
        artifact_ref: str | None = None,
    ) -> None:
        self._writer.submit(Write(
            sql="""
                INSERT INTO pipeline_events(
                    video_id, run_id, stage_id, frame_index, timestamp_ms,
                    event_type, data_json, artifact_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params=(
                video_id, run_id, stage_id, frame_index, timestamp_ms,
                event_type, json.dumps(dict(data)) if data else None, artifact_ref,
            ),
        ))

    def record_stage_finished(
        self,
        *,
        run_id: str,
        video_id: int | None,
        stage: str,
        frame_index: int,
        timestamp_ms: int,
        elapsed_ms: int,
        metadata: Mapping[str, object],
    ) -> None:
        data = {"elapsed_ms": elapsed_ms, **dict(metadata)}
        self.record(
            run_id=run_id,
            video_id=video_id,
            stage_id=stage,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            event_type="stage_finished",
            data=data,
        )

    def record_stage_metrics(
        self,
        *,
        run_id: str,
        video_id: int | None,
        stage: str,
        metrics: Mapping[str, object],
    ) -> None:
        self.record(
            run_id=run_id,
            video_id=video_id,
            stage_id=stage,
            frame_index=0,
            timestamp_ms=0,
            event_type="stage_metrics",
            data=dict(metrics),
        )

    def list_for_run(self, run_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, video_id, run_id, stage_id, frame_index, timestamp_ms, "
                "event_type, data_json, artifact_ref, created_at "
                "FROM pipeline_events WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        keys = ("id", "video_id", "run_id", "stage_id", "frame_index",
                "timestamp_ms", "event_type", "data_json", "artifact_ref",
                "created_at")
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            if d["data_json"]:
                try:
                    d["data"] = json.loads(d["data_json"])
                except Exception:
                    d["data"] = None
            out.append(d)
        return out
