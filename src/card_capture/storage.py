from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from card_capture.data.connection import open_connection
from card_capture.data.sql_queries import (
    STORAGE_CARD_INSTANCE_CANONICALS,
    STORAGE_CARD_INSTANCE_DEDUP_NO_EMBED,
    STORAGE_CARD_INSTANCE_DEDUP_WITH_EMBED,
    STORAGE_CARD_INSTANCE_FUSION_UPDATE,
    STORAGE_CARD_INSTANCE_INSERT,
    STORAGE_CARD_INSTANCES_BY_VIDEO,
    STORAGE_EVIDENCE_FRAME_INSERT,
    STORAGE_INIT_SCHEMA,
    STORAGE_PERFORMANCE_LOG_INSERT,
    STORAGE_PIPELINE_EVENT_INSERT,
    STORAGE_REVIEW_DECISION_INSERT,
    STORAGE_SAVED_CARDS_BASE,
    STORAGE_SAVED_CARD_INSERT,
    STORAGE_SAVED_CARD_REVIEW_UPDATE,
    STORAGE_SAVED_CARD_SOURCE,
    STORAGE_TRACK_TELEMETRY_INSERT,
    STORAGE_VIDEO_ID_BY_SOURCE,
    STORAGE_VIDEO_INSERT,
    STORAGE_VIDEO_INSERT_PROCESSING,
    STORAGE_VIDEO_UPDATE_STATUS,
    STORAGE_CARD_VIEW_INSERT,
    storage_alter_table_add_column,
    storage_pragma_table_info,
)
from card_capture.core.models import CardDetection, CornerDetection, PerformanceTelemetry, QualityScore


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(STORAGE_INIT_SCHEMA)
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
                STORAGE_VIDEO_INSERT,
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
            row = conn.execute(STORAGE_VIDEO_ID_BY_SOURCE, (norm, source_path)).fetchone()
            if row:
                return int(row[0])
            cursor = conn.execute(STORAGE_VIDEO_INSERT_PROCESSING, (source_path, file_hash, duration_ms, width, height))
            return int(cursor.lastrowid)

    def update_video_status(self, video_id: int, status: str) -> None:
        with self._connect() as conn:
            conn.execute(STORAGE_VIDEO_UPDATE_STATUS, (status, video_id))

    def add_performance_log(
        self, video_id: int, frame_index: int, telemetry: PerformanceTelemetry
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                STORAGE_PERFORMANCE_LOG_INSERT,
                (video_id, frame_index, telemetry.t_ingest, telemetry.t_detect, telemetry.t_refine, telemetry.t_io, telemetry.queue_wait),
            )
            return int(cursor.lastrowid)

    def add_track_telemetry(
        self, video_id: int, track_id: str, frame_index: int, polygon_area: float, aspect_ratio: float, centroid_x: float, centroid_y: float
    ) -> None:
        with self._connect() as conn:
            conn.execute(STORAGE_TRACK_TELEMETRY_INSERT, (video_id, track_id, frame_index, polygon_area, aspect_ratio, centroid_x, centroid_y))

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
            conn.execute(STORAGE_PIPELINE_EVENT_INSERT, (video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, json.dumps(data) if data else None, artifact_ref))

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
            cursor = conn.execute(STORAGE_CARD_INSTANCE_INSERT, (video_id, run_id, track_id, angle, session_id, reid_embedding))
            return int(cursor.lastrowid)

    def update_instance_deduplication(
        self, instance_id: int, visual_hash: str, duplicate_of_id: Optional[int] = None, reid_embedding: Optional[bytes] = None
    ) -> None:
        with self._connect() as conn:
            if reid_embedding is not None:
                conn.execute(STORAGE_CARD_INSTANCE_DEDUP_WITH_EMBED, (visual_hash, duplicate_of_id, reid_embedding, instance_id))
            else:
                conn.execute(STORAGE_CARD_INSTANCE_DEDUP_NO_EMBED, (visual_hash, duplicate_of_id, instance_id))

    def update_instance_fusion(self, instance_id: int, fused_path: str) -> None:
        with self._connect() as conn:
            conn.execute(STORAGE_CARD_INSTANCE_FUSION_UPDATE, (fused_path, instance_id))

    def find_canonical_for_hash(self, visual_hash: str, threshold: int = 6) -> Optional[int]:
        with self._connect() as conn:
            rows = conn.execute(STORAGE_CARD_INSTANCE_CANONICALS).fetchall()
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
            rows = conn.execute(STORAGE_CARD_INSTANCE_CANONICALS).fetchall()
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
                STORAGE_CARD_VIEW_INSERT,
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
            cursor = conn.execute(STORAGE_EVIDENCE_FRAME_INSERT, (card_view_id, source_frame_path, frame_width, frame_height, json.dumps(metrics)))
            return int(cursor.lastrowid)

    def list_card_instances(self, video_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(STORAGE_CARD_INSTANCES_BY_VIDEO, (video_id,)).fetchall()
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
            row = conn.execute(STORAGE_SAVED_CARD_SOURCE, (detection_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown detection_id/card_view id: {detection_id}")
            score_components_json = row["quality_score_json"] or "{}"
            cursor = conn.execute(
                STORAGE_SAVED_CARD_INSERT,
                (detection_id, int(row["video_id"]), image_path, final_score, row["source_path"], int(row["timestamp_ms"]), score_components_json),
            )
            return int(cursor.lastrowid)

    def set_review_decision(self, saved_card_id: int, decision: str, notes: str) -> int:
        if decision not in {"accepted", "rejected", "pending"}:
            raise ValueError("decision must be accepted, rejected, or pending")
        with self._connect() as conn:
            cursor = conn.execute(STORAGE_REVIEW_DECISION_INSERT, (saved_card_id, decision, notes))
            conn.execute(STORAGE_SAVED_CARD_REVIEW_UPDATE, (decision, saved_card_id))
            return int(cursor.lastrowid)

    def list_saved_cards(
        self, review_state: Optional[str] = None, include_duplicates: bool = False
    ) -> List[Dict[str, Any]]:
        sql = STORAGE_SAVED_CARDS_BASE
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
        return conn

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        h1 = int(hash1, 16)
        h2 = int(hash2, 16)
        return bin(h1 ^ h2).count("1")

    def _ensure_column(self, conn, table: str, column: str, ddl: str) -> None:
        rows = conn.execute(storage_pragma_table_info(table)).fetchall()
        names = {row["name"] for row in rows}
        if column not in names:
            conn.execute(storage_alter_table_add_column(table, column, ddl))
