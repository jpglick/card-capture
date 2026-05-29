"""ML repository — specialized queries for model training."""
from __future__ import annotations

from pathlib import Path

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer


class MLRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def list_fb_training_samples(self) -> list[dict]:
        """Join fb_labels with card_views to get labeled image paths."""
        with read_connection(self._db_path) as conn:
            rows = conn.execute("""
                SELECT cv.image_path, fl.side
                FROM fb_labels fl
                JOIN card_views cv ON cv.instance_id = fl.instance_id 
                                  AND cv.frame_index = fl.frame_index
                WHERE fl.side IN ('front', 'back')
            """).fetchall()
        return [{"image_path": r[0], "side": r[1]} for r in rows]

    def list_presence_training_samples(self) -> list[dict]:
        """Get labeled presence samples."""
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT image_path, label FROM presence_samples "
                "WHERE label IN ('present', 'absent')"
            ).fetchall()
        return [{"image_path": r[0], "label": r[1]} for r in rows]
