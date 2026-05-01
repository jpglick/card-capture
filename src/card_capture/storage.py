from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import CardDetection, QualityScore


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

                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL REFERENCES videos(id),
                    frame_index INTEGER NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    polygon_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    label TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    crop_path TEXT NOT NULL,
                    source_frame_path TEXT,
                    crop_width INTEGER NOT NULL,
                    crop_height INTEGER NOT NULL,
                    final_score REAL NOT NULL,
                    score_components_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS saved_cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL REFERENCES detections(id),
                    image_path TEXT NOT NULL,
                    final_score REAL NOT NULL,
                    review_state TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS review_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    saved_card_id INTEGER NOT NULL REFERENCES saved_cards(id),
                    decision TEXT NOT NULL,
                    notes TEXT NOT NULL,
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

    def add_detection(
        self,
        video_id: int,
        detection: CardDetection,
        crop_path: str,
        source_frame_path: Optional[str],
        score: QualityScore,
        crop_width: int,
        crop_height: int,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO detections (
                    video_id, frame_index, timestamp_ms, polygon_json, confidence, label,
                    metadata_json, crop_path, source_frame_path, crop_width, crop_height,
                    final_score, score_components_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    detection.frame_index,
                    detection.timestamp_ms,
                    json.dumps(detection.polygon),
                    detection.confidence,
                    detection.label,
                    json.dumps(detection.metadata),
                    crop_path,
                    source_frame_path,
                    crop_width,
                    crop_height,
                    score.total,
                    json.dumps(score.components),
                ),
            )
            return int(cursor.lastrowid)

    def add_saved_card(self, detection_id: int, image_path: str, final_score: float) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO saved_cards (detection_id, image_path, final_score)
                VALUES (?, ?, ?)
                """,
                (detection_id, image_path, final_score),
            )
            return int(cursor.lastrowid)

    def set_review_decision(self, saved_card_id: int, decision: str, notes: str) -> int:
        if decision not in {"accepted", "rejected", "pending"}:
            raise ValueError("decision must be accepted, rejected, or pending")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO review_decisions (saved_card_id, decision, notes)
                VALUES (?, ?, ?)
                """,
                (saved_card_id, decision, notes),
            )
            conn.execute(
                "UPDATE saved_cards SET review_state = ? WHERE id = ?",
                (decision, saved_card_id),
            )
            return int(cursor.lastrowid)

    def list_saved_cards(self, review_state: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = """
            SELECT
                saved_cards.id,
                saved_cards.detection_id,
                saved_cards.image_path,
                saved_cards.final_score,
                saved_cards.review_state,
                videos.source_path,
                detections.timestamp_ms,
                detections.score_components_json
            FROM saved_cards
            JOIN detections ON detections.id = saved_cards.detection_id
            JOIN videos ON videos.id = detections.video_id
        """
        params = ()
        if review_state is not None:
            sql += " WHERE saved_cards.review_state = ?"
            params = (review_state,)
        sql += " ORDER BY saved_cards.final_score DESC, saved_cards.id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "detection_id": int(row["detection_id"]),
                "image_path": row["image_path"],
                "final_score": float(row["final_score"]),
                "review_state": row["review_state"],
                "source_path": row["source_path"],
                "timestamp_ms": int(row["timestamp_ms"]),
                "score_components": json.loads(row["score_components_json"]),
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
