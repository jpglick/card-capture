"""Service layer for pipeline run management.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, List, Optional


class RunService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def list_runs(self, video_id: Optional[int] = None) -> List[dict[str, Any]]:
        """Return a list of pipeline runs, newest first."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            params: list = []
            where = "WHERE 1=1"
            if video_id:
                where += " AND pr.video_id = ?"
                params.append(video_id)
            rows = conn.execute(f"""
                SELECT pr.run_id, pr.video_id, pr.status, pr.cards_extracted,
                       pr.started_at as created_at, pr.finished_at,
                       v.source_path
                FROM pipeline_runs pr
                LEFT JOIN videos v ON v.id = pr.video_id
                {where}
                ORDER BY pr.started_at DESC
            """, params).fetchall()

            runs = []
            for r in rows:
                started = r["created_at"] or ""
                finished = r["finished_at"] or ""
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
                    "run_id": r["run_id"],
                    "video_id": Path(r["source_path"]).stem if r["source_path"] else str(r["video_id"]),
                    "status": r["status"],
                    "cards_extracted": r["cards_extracted"],
                    "elapsed_ms": elapsed_ms,
                    "created_at": started,
                })
            return runs

    def get_run_details(self, run_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full details for a run."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT pr.run_id, pr.video_id, pr.status, pr.cards_extracted,
                       pr.started_at, pr.finished_at,
                       pr.detect_telemetry_json,
                       v.source_path, v.duration_ms as video_duration_ms
                FROM pipeline_runs pr
                LEFT JOIN videos v ON v.id = pr.video_id
                WHERE pr.run_id = ?
                """,
                (run_id,),
            ).fetchone()

            if not row:
                return None

            events = conn.execute(
                "SELECT event_type, data_json, created_at FROM pipeline_events "
                "WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()

            # Fetch persisted log lines (last 50)
            try:
                log_rows = conn.execute(
                    "SELECT line FROM pipeline_run_logs WHERE run_id = ? "
                    "ORDER BY id ASC",
                    (run_id,),
                ).fetchall()
                logs = [r["line"] for r in log_rows][-50:]
            except Exception:
                logs = []

            # Extract stage timings from pipeline_events
            stage_timings = []
            import json as _json
            for ev in events:
                et = ev["event_type"] or ""
                if et.startswith("stage_"):
                    stage = et[len("stage_"):]
                    elapsed = None
                    if ev["data_json"]:
                        try:
                            elapsed = _json.loads(ev["data_json"]).get("elapsed_ms")
                        except Exception:
                            pass
                    if elapsed is not None:
                        stage_timings.append({"stage": stage, "elapsed_ms": elapsed})

            started = row["started_at"] or ""
            finished = row["finished_at"] or ""
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

            detect_telemetry = None
            if row["detect_telemetry_json"]:
                try:
                    import json as _json2
                    detect_telemetry = _json2.loads(row["detect_telemetry_json"])
                except Exception:
                    pass

            return {
                "run_id": run_id,
                "video_id": Path(row["source_path"]).stem if row["source_path"] else str(row["video_id"]),
                "status": row["status"],
                "cards_extracted": row["cards_extracted"],
                "elapsed_ms": elapsed_ms,
                "created_at": started,
                "events": [dict(e) for e in events],
                "logs": logs,
                "stage_timings": stage_timings,
                "video_duration_ms": row["video_duration_ms"],
                "detect_telemetry": detect_telemetry,
            }

    def get_run_resources(self, run_id: str) -> dict:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            run = conn.execute(
                "SELECT started_at, finished_at, host_info_json FROM pipeline_runs WHERE run_id = ?",
                (run_id,)
            ).fetchone()
            if not run:
                return {}

            samples = conn.execute(
                "SELECT elapsed_s, cpu_pct, mem_used_mb, mem_pct, gpu_pct, vram_used_mb "
                "FROM run_resource_samples WHERE run_id = ? ORDER BY elapsed_s",
                (run_id,)
            ).fetchall()

            # Stage markers from pipeline_events (if table exists)
            stage_markers = []
            if run["started_at"]:
                try:
                    from datetime import datetime
                    fmt = "%Y-%m-%d %H:%M:%S"
                    run_start = datetime.strptime(run["started_at"], fmt)
                    events = conn.execute(
                        "SELECT stage_id, event_type, created_at FROM pipeline_events "
                        "WHERE run_id = ? AND event_type LIKE 'stage_%' ORDER BY created_at",
                        (run_id,)
                    ).fetchall()
                    for ev in events:
                        if ev["created_at"]:
                            try:
                                ev_ts = datetime.strptime(ev["created_at"], fmt)
                                elapsed_s = (ev_ts - run_start).total_seconds()
                                stage_markers.append({
                                    "name": ev["stage_id"] or ev["event_type"],
                                    "elapsed_s": elapsed_s,
                                })
                            except Exception:
                                pass
                except Exception:
                    pass

            import json as _json
            host_info = None
            if run["host_info_json"]:
                try:
                    host_info = _json.loads(run["host_info_json"])
                except Exception:
                    pass

            return {
                "run_id": run_id,
                "host_info": host_info,
                "samples": [dict(s) for s in samples],
                "stage_markers": stage_markers,
            }
