from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import CornerDetection


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'processing',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS card_instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL REFERENCES videos(id),
                    track_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS card_views (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_instance_id INTEGER NOT NULL REFERENCES card_instances(id),
                    frame_index INTEGER NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    corners_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rectified_path TEXT,
                    quality_score_json TEXT,
                    is_canonical INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS evidence_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_view_id INTEGER NOT NULL REFERENCES card_views(id),
                    source_frame_path TEXT NOT NULL,
                    frame_width INTEGER NOT NULL,
                    frame_height INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def add_video(
        self,
        source_path: str,
        file_hash: str,
        duration_ms: int,
        width: int,
        height: int,
        status: str = "processing",
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO videos (source_path, file_hash, duration_ms, width, height, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_path, file_hash, duration_ms, width, height, status),
            )
            return int(cursor.lastrowid)

    def update_video_status(self, video_id: int, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))

    def add_card_instance(
        self,
        video_id: int,
        track_id: str,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO card_instances (video_id, track_id)
                VALUES (?, ?)
                """,
                (video_id, track_id),
            )
            return int(cursor.lastrowid)

    def add_card_view(
        self,
        card_instance_id: int,
        frame_index: int,
        timestamp_ms: int,
        detection: CornerDetection,
        rectified_path: Optional[str] = None,
        quality_score: Optional[Dict[str, float]] = None,
        is_canonical: bool = False,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO card_views (
                    card_instance_id,
                    frame_index,
                    timestamp_ms,
                    corners_json,
                    confidence,
                    rectified_path,
                    quality_score_json,
                    is_canonical,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    card_instance_id,
                    frame_index,
                    timestamp_ms,
                    json.dumps(detection.corners),
                    detection.confidence,
                    rectified_path,
                    json.dumps(quality_score) if quality_score is not None else None,
                    int(is_canonical),
                    json.dumps(detection.metadata),
                ),
            )
            return int(cursor.lastrowid)

    def add_evidence_frame(
        self,
        card_view_id: int,
        source_frame_path: str,
        frame_width: int,
        frame_height: int,
        metrics: Dict[str, float],
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO evidence_frames (
                    card_view_id, source_frame_path, frame_width, frame_height, metrics_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (card_view_id, source_frame_path, frame_width, frame_height, json.dumps(metrics)),
            )
            return int(cursor.lastrowid)

    def list_card_instances(self, video_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, video_id, track_id, created_at, updated_at
                FROM card_instances
                WHERE video_id = ?
                ORDER BY id ASC
                """,
                (video_id,),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "video_id": int(row["video_id"]),
                "track_id": row["track_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
