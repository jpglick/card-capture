"""labeling repository for fb_labels and truth_files."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class LabelingRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def store_fb_label(self, instance_id: str, frame_index: int, side: str, labeler: str = "human", source_run_id: int | None = None) -> None:
        self._writer.submit(Write(
            sql="""
                INSERT INTO fb_labels(source_run_id, instance_id, frame_index, side, labeler)
                VALUES (?, ?, ?, ?, ?)
            """,
            params=(source_run_id, instance_id, frame_index, side, labeler),
        ))

    def list_for_instance(self, instance_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT label_id, source_run_id, instance_id, frame_index, side, labeler, created_at "
                "FROM fb_labels WHERE instance_id=? ORDER BY created_at DESC",
                (instance_id,),
            ).fetchall()
        keys = ("label_id", "source_run_id", "instance_id", "frame_index", "side", "labeler", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    def store_truth_payload(self, video_id: str, payload: Mapping[str, object], schema_version: int = 1) -> None:
        self._writer.submit(Write(
            sql="""
                INSERT OR REPLACE INTO truth_files(video_id, schema_version, payload_json)
                VALUES (?, ?, ?)
            """,
            params=(video_id, schema_version, json.dumps(dict(payload))),
        ))

    def get_truth_payload(self, video_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT payload_json FROM truth_files WHERE video_id=?",
                (video_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def list_unlabeled(self, limit: int = 50) -> list[dict]:
        # Implementation depends on exact definitions of unlabeled.
        # This is a placeholder as requested in the plan: `list_unlabeled(limit)`.
        with read_connection(self._db_path) as conn:
            # We assume instances without any label.
            rows = conn.execute(
                """
                SELECT instance_id, video_id, fused_image_path 
                FROM card_instances 
                WHERE instance_id NOT IN (SELECT instance_id FROM fb_labels)
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        keys = ("instance_id", "video_id", "fused_image_path")
        return [dict(zip(keys, r)) for r in rows]
