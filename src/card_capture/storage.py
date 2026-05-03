from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import CardDetection, CornerDetection, PerformanceTelemetry, QualityScore


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
                    visual_hash TEXT,
                    is_duplicate_of INTEGER REFERENCES card_instances(id),
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

                -- Backward-compatibility surface for review + legacy tests.
                CREATE TABLE IF NOT EXISTS saved_cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    video_id INTEGER NOT NULL REFERENCES videos(id),
                    image_path TEXT NOT NULL,
                    final_score REAL NOT NULL,
                    review_state TEXT NOT NULL DEFAULT 'pending',
                    source_path TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    score_components_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS review_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    saved_card_id INTEGER NOT NULL REFERENCES saved_cards(id),
                    decision TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS performance_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL REFERENCES videos(id),
                    frame_index INTEGER NOT NULL,
                    t_ingest REAL NOT NULL,
                    t_detect REAL NOT NULL,
                    t_refine REAL NOT NULL,
                    t_io REAL NOT NULL,
                    queue_wait REAL NOT NULL,
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

    def add_performance_log(
        self, video_id: int, frame_index: int, telemetry: PerformanceTelemetry
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO performance_logs (
                    video_id, frame_index, t_ingest, t_detect, t_refine, t_io, queue_wait
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    frame_index,
                    telemetry.t_ingest,
                    telemetry.t_detect,
                    telemetry.t_refine,
                    telemetry.t_io,
                    telemetry.queue_wait,
                ),
            )
            return int(cursor.lastrowid)

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

    def update_instance_deduplication(
        self, instance_id: int, visual_hash: str, duplicate_of_id: Optional[int] = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE card_instances SET visual_hash = ?, is_duplicate_of = ? WHERE id = ?",
                (visual_hash, duplicate_of_id, instance_id),
            )

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
                SELECT id, video_id, track_id, visual_hash, is_duplicate_of, created_at, updated_at
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
                "visual_hash": row["visual_hash"],
                "is_duplicate_of": row["is_duplicate_of"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    # Compatibility layer for legacy pipeline/review call sites.
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
        corner_detection = CornerDetection(
            corners=detection.polygon,
            confidence=detection.confidence,
            metadata=detection.metadata,
        )
        instance_id = self.add_card_instance(
            video_id=video_id,
            track_id=f"legacy_{detection.frame_index}_{detection.timestamp_ms}",
        )
        view_id = self.add_card_view(
            card_instance_id=instance_id,
            frame_index=detection.frame_index,
            timestamp_ms=detection.timestamp_ms,
            detection=corner_detection,
            rectified_path=crop_path,
            quality_score=score.components,
            is_canonical=False,
        )
        self.add_evidence_frame(
            card_view_id=view_id,
            source_frame_path=source_frame_path or crop_path,
            frame_width=crop_width,
            frame_height=crop_height,
            metrics={"legacy_score_total": float(score.total)},
        )
        return view_id

    def add_saved_card(self, detection_id: int, image_path: str, final_score: float) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    card_views.card_instance_id,
                    card_views.timestamp_ms,
                    card_views.quality_score_json,
                    videos.id AS video_id,
                    videos.source_path
                FROM card_views
                JOIN card_instances ON card_instances.id = card_views.card_instance_id
                JOIN videos ON videos.id = card_instances.video_id
                WHERE card_views.id = ?
                """,
                (detection_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown detection_id/card_view id: {detection_id}")
            score_components_json = row["quality_score_json"] or "{}"
            cursor = conn.execute(
                """
                INSERT INTO saved_cards (
                    detection_id,
                    video_id,
                    image_path,
                    final_score,
                    review_state,
                    source_path,
                    timestamp_ms,
                    score_components_json
                )
                VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    detection_id,
                    int(row["video_id"]),
                    image_path,
                    final_score,
                    row["source_path"],
                    int(row["timestamp_ms"]),
                    score_components_json,
                ),
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
                """
                UPDATE saved_cards
                SET review_state = ?
                WHERE id = ?
                """,
                (decision, saved_card_id),
            )
            return int(cursor.lastrowid)

    def list_saved_cards(self, review_state: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = """
            SELECT
                id,
                detection_id,
                image_path,
                final_score,
                review_state,
                source_path,
                timestamp_ms,
                score_components_json
            FROM saved_cards
        """
        params: tuple[Any, ...] = ()
        if review_state is not None:
            sql += " WHERE review_state = ?"
            params = (review_state,)
        sql += " ORDER BY final_score DESC, id ASC"
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
