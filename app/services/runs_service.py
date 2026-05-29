"""Service layer for pipeline run management."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from card_capture.data.connection import read_connection
from card_capture.data.sql_queries import (
    RUN_DETAILS,
    RUN_EVENTS,
    RUN_LOGS,
    RUN_RESOURCE_RANGE,
    RUN_RESOURCE_SAMPLES,
    RUN_STAGE_EVENTS,
    RUNS_LIST_BASE,
)


class RunService:
    def __init__(self, db_path: Path, runs_repo=None, events_repo=None) -> None:
        self.db_path = db_path
        self._runs_repo = runs_repo
        self._events_repo = events_repo

    def list_runs(self, video_id: Optional[int] = None) -> List[dict[str, Any]]:
        """Return a list of pipeline runs, newest first."""
        with read_connection(self.db_path) as conn:
            params: list = []
            where = "WHERE 1=1"
            if video_id:
                where += " AND pr.video_id = ?"
                params.append(video_id)
            rows = conn.execute(RUNS_LIST_BASE.format(where=where), params).fetchall()

            runs = []
            for r in rows:
                run_id, vid, status, extracted, started, finished, source_path = r
                elapsed_ms = 0
                if started and finished:
                    from datetime import datetime
                    fmt = "%Y-%m-%d %H:%M:%S"
                    try:
                        elapsed_ms = int(
                            (datetime.strptime(finished, fmt) - datetime.strptime(started, fmt))
                            .total_seconds() * 1000
                        )
                    except ValueError:
                        pass
                runs.append({
                    "run_id": run_id,
                    "video_id": Path(source_path).stem if source_path else str(vid),
                    "status": status,
                    "cards_extracted": extracted,
                    "elapsed_ms": elapsed_ms,
                    "created_at": started,
                })
            return runs

    def get_run_details(self, run_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full details for a run."""
        with read_connection(self.db_path) as conn:
            row = conn.execute(RUN_DETAILS, (run_id,)).fetchone()

            if not row:
                return None

            events = conn.execute(RUN_EVENTS, (run_id,)).fetchall()

            # Fetch persisted log lines (last 50)
            try:
                log_rows = conn.execute(RUN_LOGS, (run_id,)).fetchall()
                logs = [r[0] for r in log_rows][-50:]
            except Exception:
                logs = []

            # Extract stage timings from pipeline_events
            stage_timings = []
            for ev in events:
                et = ev[0] or ""
                if et.startswith("stage_"):
                    stage = et[len("stage_"):]
                    elapsed = None
                    if ev[1]:
                        try:
                            elapsed = json.loads(ev[1]).get("elapsed_ms")
                        except Exception:
                            pass
                    if elapsed is not None:
                        stage_timings.append({"stage": stage, "elapsed_ms": elapsed})

            run_id_db, video_id, status, cards, started, finished, source_path, video_duration = row
            elapsed_ms = 0
            if started and finished:
                from datetime import datetime
                fmt = "%Y-%m-%d %H:%M:%S"
                try:
                    elapsed_ms = int(
                        (datetime.strptime(finished, fmt) - datetime.strptime(started, fmt))
                        .total_seconds() * 1000
                    )
                except ValueError:
                    pass

            return {
                "run_id": run_id,
                "video_id": Path(source_path).stem if source_path else str(video_id),
                "status": status,
                "cards_extracted": cards,
                "elapsed_ms": elapsed_ms,
                "created_at": started,
                "events": [{"event_type": e[0], "data_json": e[1], "created_at": e[2]} for e in events],
                "logs": logs,
                "stage_timings": stage_timings,
                "video_duration_ms": video_duration,
            }

    def get_run_resources(self, run_id: str) -> dict:
        with read_connection(self.db_path) as conn:
            row = conn.execute(RUN_RESOURCE_RANGE, (run_id,)).fetchone()
            if not row:
                return {}

            started, finished = row
            samples = conn.execute(RUN_RESOURCE_SAMPLES, (run_id,)).fetchall()

            # Stage markers from pipeline_events (if table exists)
            stage_markers = []
            if started:
                try:
                    from datetime import datetime
                    fmt = "%Y-%m-%d %H:%M:%S"
                    run_start = datetime.strptime(started, fmt)
                    events = conn.execute(RUN_STAGE_EVENTS, (run_id,)).fetchall()
                    for ev in events:
                        if ev[2]:
                            try:
                                ev_ts = datetime.strptime(ev[2], fmt)
                                elapsed_s = (ev_ts - run_start).total_seconds()
                                stage_markers.append({
                                    "name": ev[0] or ev[1],
                                    "elapsed_s": elapsed_s,
                                })
                            except Exception:
                                pass
                except Exception:
                    pass

            return {
                "run_id": run_id,
                "samples": [
                    {
                        "elapsed_s": s[0],
                        "cpu_pct": s[1],
                        "mem_used_mb": s[2],
                        "mem_pct": s[3],
                        "gpu_pct": s[4],
                        "vram_used_mb": s[5],
                    }
                    for s in samples
                ],
                "stage_markers": stage_markers,
            }
