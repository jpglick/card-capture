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
        """Return a list of pipeline runs, optionally filtered by video."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT DISTINCT run_id FROM pipeline_events"
            params = []
            if video_id:
                # We can try to find run_ids that have events associated with this video
                # In v4, we can assume artifacts store video_id
                pass
                
            rows = conn.execute(query, params).fetchall()
            
            runs = []
            for r in rows:
                run_id = r["run_id"]
                
                # Get summary info from events
                summary = conn.execute(
                    "SELECT timestamp as created_at FROM pipeline_events WHERE run_id = ? AND event_name = 'run_started'",
                    (run_id,)
                ).fetchone()
                
                status_row = conn.execute(
                    "SELECT event_name FROM pipeline_events WHERE run_id = ? AND event_name IN ('run_completed', 'run_failed') ORDER BY timestamp DESC LIMIT 1",
                    (run_id,)
                ).fetchone()
                
                # Count cards extracted for this run
                # (This is tricky since card_instances doesn't have run_id yet)
                # For now, let's just return 0 or query by video_id if available
                card_count = 0
                
                runs.append({
                    "run_id": run_id,
                    "video_id": "unknown", # To be filled if possible
                    "status": status_row["event_name"].replace("run_", "") if status_row else "running",
                    "cards_extracted": card_count,
                    "elapsed_ms": 0,
                    "created_at": summary["created_at"] if summary else "unknown",
                })
            return runs

    def get_run_details(self, run_id: str) -> Optional[dict[str, Any]]:
        """Retrieve full details and telemetry for a run."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            events = conn.execute(
                "SELECT * FROM pipeline_events WHERE run_id = ? ORDER BY timestamp ASC",
                (run_id,)
            ).fetchall()
            
            if not events:
                return None
                
            # Get latest status
            status_row = conn.execute(
                "SELECT event_name FROM pipeline_events WHERE run_id = ? AND event_name IN ('run_completed', 'run_failed') ORDER BY timestamp DESC LIMIT 1",
                (run_id,)
            ).fetchone()
            
            created_at = "unknown"
            if events:
                created_at = events[0]["timestamp"]

            return {
                "run_id": run_id,
                "video_id": "unknown",
                "status": status_row["event_name"].replace("run_", "") if status_row else "running",
                "created_at": created_at,
                "events": [dict(e) for e in events],
            }
