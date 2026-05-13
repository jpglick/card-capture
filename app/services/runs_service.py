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
            
            # Select distinct runs. We use COALESCE to handle legacy null run_ids.
            query = """
                SELECT 
                    COALESCE(run_id, 'legacy-' || video_id) as run_id,
                    video_id,
                    MIN(created_at) as created_at
                FROM pipeline_events
                GROUP BY 1
                ORDER BY created_at DESC
            """
            params = []
            if video_id:
                query = """
                    SELECT 
                        COALESCE(run_id, 'legacy-' || video_id) as run_id,
                        video_id,
                        MIN(created_at) as created_at
                    FROM pipeline_events
                    WHERE video_id = ?
                    GROUP BY 1
                    ORDER BY created_at DESC
                """
                params.append(video_id)
                
            rows = conn.execute(query, params).fetchall()
            
            runs = []
            for r in rows:
                run_id = r["run_id"]
                vid = r["video_id"]
                
                # Get latest status for this run
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
                
                # Count cards extracted for this run
                card_count = conn.execute(
                    "SELECT COUNT(*) FROM card_instances WHERE run_id = ? OR (run_id IS NULL AND 'legacy-' || video_id = ?)",
                    (run_id, run_id)
                ).fetchone()[0]
                
                # Get video filename
                video_row = conn.execute("SELECT source_path FROM videos WHERE id = ?", (vid,)).fetchone()
                video_id_str = Path(video_row["source_path"]).stem if video_row else str(vid)
                
                runs.append({
                    "run_id": run_id,
                    "video_id": video_id_str,
                    "status": status_row["event_type"].replace("run_", "") if status_row else "running",
                    "cards_extracted": card_count,
                    "elapsed_ms": 0, # To be computed if start/end events exist
                    "created_at": r["created_at"],
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
