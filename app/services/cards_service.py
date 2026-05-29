"""Service layer for card management."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from card_capture.data.connection import read_connection
from card_capture.data.sql_queries import (
    CARD_GET_ONE,
    CARD_SOURCE_FRAMES,
    CARDS_CANONICAL_VIEW_WITH_FRAME,
    CARDS_CANONICAL_VIEW,
    CARDS_COUNT_BASE,
    CARDS_LIST_BASE,
    CARDS_VIDEO_SOURCE,
)


def _to_file_url(path: Optional[str]) -> Optional[str]:
    """Convert an absolute or relative filesystem path to a /files/ URL.

    The FastAPI app mounts card_capture_output/ at /files/. Works for both
    relative paths (card_capture_output/...) and absolute paths that contain
    card_capture_output somewhere in the path.
    """
    if not path:
        return None
    p = Path(path)
    # Relative path already rooted at card_capture_output
    try:
        rel = p.relative_to("card_capture_output")
        return f"/files/{rel}"
    except ValueError:
        pass
    # Absolute path: find card_capture_output in parts and take the tail
    parts = p.parts
    try:
        idx = parts.index("card_capture_output")
        rel = "/".join(parts[idx + 1:])
        return f"/files/{rel}" if rel else None
    except ValueError:
        pass
    return None


class CardService:
    def __init__(self, db_path: Path, cards_repo=None) -> None:
        self.db_path = db_path
        self._repo = cards_repo

    def list_cards(
        self,
        run_id: Optional[str] = None,
        video_id: Optional[int] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """Return a paginated list of extracted card instances."""
        query = CARDS_LIST_BASE
        count_query = CARDS_COUNT_BASE
        params = []
        where_clauses = ["hidden = 0"]

        if run_id:
            where_clauses.append("(run_id = ? OR (run_id IS NULL AND 'legacy-' || video_id = ?))")
            params.extend([run_id, run_id])
        if video_id:
            where_clauses.append("video_id = ?")
            params.append(video_id)

        if where_clauses:
            clause = " WHERE " + " AND ".join(where_clauses)
            query += clause
            count_query += clause
            
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        
        with read_connection(str(self.db_path)) as conn:
            total = conn.execute(count_query, params).fetchone()[0]
            
            # Add limit and offset
            rows_params = list(params)
            rows_params.extend([page_size, (page - 1) * page_size])
            rows = conn.execute(query, rows_params).fetchall()
            
            items = []
            for r in rows:
                c_id, track_id, vid, r_id, angle, fused, created = r
                
                # Get the canonical view to get confidence
                view = conn.execute(
                    CARDS_CANONICAL_VIEW,
                    (c_id,)
                ).fetchone()
                
                video_row = conn.execute(CARDS_VIDEO_SOURCE, (vid,)).fetchone()
                video_id_str = Path(video_row[0]).stem if video_row else str(vid)

                items.append({
                    "card_id": track_id,
                    "instance_id": str(c_id),
                    "video_id": video_id_str,
                    "run_id": r_id or f"legacy-{vid}",
                    "side": angle or "Front",
                    "is_foil": False,
                    "confidence": view[0] if view else 0.0,
                    "review_state": "pending",
                    "canonical_url": _to_file_url(view[1] if view else None),
                    "fused_url": _to_file_url(fused),
                    "created_at": created,
                })
            
            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "items": items,
            }

    def get_card(self, card_instance_id: int) -> Optional[dict[str, Any]]:
        """Retrieve a single card record by internal database ID."""
        with read_connection(str(self.db_path)) as conn:
            row = conn.execute(CARD_GET_ONE, (card_instance_id,)).fetchone()
            if not row:
                return None
            
            c_id, track_id, vid, r_id, angle, fused, created = row
            
            view = conn.execute(
                CARDS_CANONICAL_VIEW_WITH_FRAME,
                (c_id,)
            ).fetchone()

            source_frames = conn.execute(CARD_SOURCE_FRAMES, (c_id,)).fetchall()
            
            video_row = conn.execute(CARDS_VIDEO_SOURCE, (vid,)).fetchone()
            video_id_str = Path(video_row[0]).stem if video_row else str(vid)

            return {
                "card_id": track_id,
                "instance_id": str(c_id),
                "video_id": video_id_str,
                "run_id": r_id or f"legacy-{vid}",
                "side": angle or "front",
                "is_foil": False,
                "confidence": view[0] if view else 0.0,
                "review_state": "pending",
                "canonical_url": _to_file_url(view[1] if view else None),
                "fused_url": _to_file_url(fused),
                "created_at": created,
                "source_frame_indices": [f[0] for f in source_frames],
                "quality_score": {
                    "total": view[0] if view else 0.0, # Placeholder
                }
            }
