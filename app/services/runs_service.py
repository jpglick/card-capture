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
        """Retrieve full details and telemetry for a run."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            events = conn.execute(
                """
                SELECT * FROM pipeline_events 
                WHERE (run_id = ? OR (run_id IS NULL AND 'legacy-' || video_id = ?))
                ORDER BY created_at ASC
                """,
                (run_id, run_id)
            ).fetchall()
            
            if not events:
                return None
            
            # Get latest status
            status_row = conn.execute(
                """
                SELECT event_type 
                FROM pipeline_events 
                WHERE (run_id = ? OR (run_id IS NULL AND 'legacy-' || video_id = ?))
                  AND event_type IN ('run_completed', 'run_failed') 
                ORDER BY created_at DESC LIMIT 1
                """,
                (run_id, run_id)
            ).fetchone()
            
            vid = events[0]["video_id"]
            video_row = conn.execute("SELECT source_path FROM videos WHERE id = ?", (vid,)).fetchone()
            video_id_str = Path(video_row["source_path"]).stem if video_row else str(vid)

            return {
                "run_id": run_id,
                "video_id": video_id_str,
                "status": status_row["event_type"].replace("run_", "") if status_row else "running",
                "created_at": events[0]["created_at"],
                "events": [dict(e) for e in events],
            }
