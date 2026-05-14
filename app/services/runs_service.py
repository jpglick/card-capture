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
                SELECT pr.*, v.source_path
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

            return {
                "run_id": run_id,
                "video_id": Path(row["source_path"]).stem if row["source_path"] else str(row["video_id"]),
                "status": row["status"],
                "cards_extracted": row["cards_extracted"],
                "elapsed_ms": elapsed_ms,
                "created_at": started,
                "events": [dict(e) for e in events],
            }
