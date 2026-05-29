from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from card_capture.data.connection import open_connection
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
                    run_id TEXT,
                    track_id TEXT NOT NULL,
                    session_id TEXT,
                    visual_hash TEXT,
                    reid_embedding BLOB,
                    is_duplicate_of INTEGER REFERENCES card_instances(id),
                    angle TEXT,
                    fused_image_path TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id, track_id)
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
                    glare_x REAL,
                    glare_y REAL,
                    sharpness REAL,
                    glare_mask_b64 TEXT,
                    laplacian_heatmap_b64 TEXT,
                    initial_confidence REAL,
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

                CREATE TABLE IF NOT EXISTS track_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL REFERENCES videos(id),
                    track_id TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    polygon_area REAL NOT NULL,
                    aspect_ratio REAL NOT NULL,
                    centroid_x REAL NOT NULL,
                    centroid_y REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS pipeline_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL REFERENCES videos(id),
                    run_id TEXT,
                    stage_id TEXT,
                    frame_index INTEGER NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT,
                    artifact_ref TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._ensure_column(conn, "card_instances", "angle", "TEXT")
            self._ensure_column(conn, "card_instances", "session_id", "TEXT")
            self._ensure_column(conn, "card_instances", "run_id", "TEXT")
            self._ensure_column(conn, "card_instances", "fused_image_path", "TEXT")
            self._ensure_column(conn, "card_instances", "reid_embedding", "BLOB")
            self._ensure_column(conn, "pipeline_events", "run_id", "TEXT")
            self._ensure_column(conn, "pipeline_events", "stage_id", "TEXT")
            self._ensure_column(conn, "pipeline_events", "artifact_ref", "TEXT")
            self._ensure_column(conn, "card_views", "glare_x", "REAL")
            self._ensure_column(conn, "card_views", "glare_y", "REAL")
            self._ensure_column(conn, "card_views", "sharpness", "REAL")
            self._ensure_column(conn, "card_views", "glare_mask_b64", "TEXT")
            self._ensure_column(conn, "card_views", "laplacian_heatmap_b64", "TEXT")
            self._ensure_column(conn, "card_views", "initial_confidence", "REAL")

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

    def get_or_create_video(
        self,
        source_path: str,
        file_hash: str,
        duration_ms: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> int:
        """Return the existing video ID for source_path, or insert a new row.

        Normalises to absolute path before lookup so relative and absolute
        references to the same file don't create duplicate rows.
        """
        norm = str(Path(source_path).resolve())
        with self._connect() as conn:
            # Check both the normalised absolute path and the original value
            row = conn.execute(
                "SELECT id FROM videos WHERE source_path = ? OR source_path = ?",
                (norm, source_path),
            ).fetchone()
            if row:
                return int(row[0])
            cursor = conn.execute(
                "INSERT INTO videos (source_path, file_hash, duration_ms, width, height, status) "
                "VALUES (?, ?, ?, ?, ?, 'processing')",
                (source_path, file_hash, duration_ms, width, height),
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

    def add_track_telemetry(
        self, video_id: int, track_id: str, frame_index: int, polygon_area: float, aspect_ratio: float, centroid_x: float, centroid_y: float
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO track_telemetry (video_id, track_id, frame_index, polygon_area, aspect_ratio, centroid_x, centroid_y)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (video_id, track_id, frame_index, polygon_area, aspect_ratio, centroid_x, centroid_y)
            )

    def add_pipeline_event(
        self, 
        video_id: int, 
        frame_index: int, 
        timestamp_ms: int, 
        event_type: str, 
        data: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        artifact_ref: Optional[str] = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_events (video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, data_json, artifact_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, json.dumps(data) if data else None, artifact_ref),
            )

    def add_card_instance(
        self,
        video_id: int,
        track_id: str,
        angle: Optional[str] = None,
        session_id: Optional[str] = None,
        reid_embedding: Optional[bytes] = None,
        run_id: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO card_instances (video_id, run_id, track_id, angle, session_id, reid_embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (video_id, run_id, track_id, angle, session_id, reid_embedding),
            )
            return int(cursor.lastrowid)

    def update_instance_deduplication(
        self, instance_id: int, visual_hash: str, duplicate_of_id: Optional[int] = None, reid_embedding: Optional[bytes] = None
    ) -> None:
        with self._connect() as conn:
            if reid_embedding is not None:
                conn.execute(
                    """
                    UPDATE card_instances
                    SET visual_hash = ?, is_duplicate_of = ?, reid_embedding = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (visual_hash, duplicate_of_id, reid_embedding, instance_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE card_instances
                    SET visual_hash = ?, is_duplicate_of = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (visual_hash, duplicate_of_id, instance_id),
                )

    def update_instance_fusion(self, instance_id: int, fused_path: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE card_instances
                SET fused_image_path = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (fused_path, instance_id),
            )

    def find_canonical_for_hash(self, visual_hash: str, threshold: int = 6) -> Optional[int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, visual_hash
                FROM card_instances
                WHERE visual_hash IS NOT NULL AND is_duplicate_of IS NULL
                """
            ).fetchall()
        for row in rows:
            if self._hamming_distance(visual_hash, row["visual_hash"]) <= threshold:
                return int(row["id"])
        return None

    def find_canonical_for_hashes(
        self, visual_hashes: list[str], threshold: int = 6
    ) -> Optional[int]:
        if not visual_hashes:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, visual_hash
                FROM card_instances
                WHERE visual_hash IS NOT NULL AND is_duplicate_of IS NULL
                """
            ).fetchall()
        best_id: Optional[int] = None
        best_dist: Optional[int] = None
        for row in rows:
            d = min(
                self._hamming_distance(candidate, row["visual_hash"])
                for candidate in visual_hashes
            )
            if best_dist is None or d < best_dist:
                best_dist = d
                best_id = int(row["id"])
        if best_dist is not None and best_dist <= threshold:
            return best_id
        return None

    def add_card_view(
        self,
        card_instance_id: int,
        frame_index: int,
        timestamp_ms: int,
        detection: CornerDetection,
        rectified_path: Optional[str] = None,
        quality_score: Optional[Dict[str, float]] = None,
        is_canonical: bool = False,
        glare_x: Optional[float] = None,
        glare_y: Optional[float] = None,
        sharpness: Optional[float] = None,
        glare_mask: Optional[bytes] = None,
        laplacian_heatmap: Optional[bytes] = None,
        initial_confidence: Optional[float] = None,
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
                    glare_x,
                    glare_y,
                    sharpness,
                    glare_mask_b64,
                    laplacian_heatmap_b64,
                    initial_confidence,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    glare_x,
                    glare_y,
                    sharpness,
                    base64.b64encode(glare_mask).decode("ascii") if glare_mask else None,
                    base64.b64encode(laplacian_heatmap).decode("ascii")
                    if laplacian_heatmap
                    else None,
                    initial_confidence,
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
                SELECT
                    id,
                    video_id,
                    track_id,
                    visual_hash,
                    is_duplicate_of,
                    angle,
                    fused_image_path,
                    created_at,
                    updated_at
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
                "angle": row["angle"],
                "fused_image_path": row["fused_image_path"],
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

    def list_saved_cards(
        self, review_state: Optional[str] = None, include_duplicates: bool = False
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT
                sc.id,
                sc.detection_id,
                sc.image_path,
                sc.final_score,
                sc.review_state,
                sc.source_path,
                sc.timestamp_ms,
                sc.score_components_json
            FROM saved_cards sc
            JOIN card_views cv ON cv.id = sc.detection_id
            JOIN card_instances ci ON ci.id = cv.card_instance_id
        """
        params: list[Any] = []
        conditions: list[str] = []
        if not include_duplicates:
            conditions.append("ci.is_duplicate_of IS NULL")
        if review_state is not None:
            conditions.append("sc.review_state = ?")
            params.append(review_state)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY sc.final_score DESC, sc.id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
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

    def _connect(self):
        conn = open_connection(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        h1 = int(hash1, 16)
        h2 = int(hash2, 16)
        return bin(h1 ^ h2).count("1")

    def _ensure_column(self, conn, table: str, column: str, ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        names = {row["name"] for row in rows}
        if column not in names:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
