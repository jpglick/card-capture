"""Cards repository.

Writes to the production `card_instances` table (extended with front_crop /
back_crop columns by migration 0013) and the v5.5 `card_view_metrics` table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write


class CardsRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def store_final_cards(
        self,
        *,
        run_id: str,
        video_id: int,
        cards: Iterable[Mapping[str, Any]],
    ) -> None:
        for c in cards:
            get = c.get if isinstance(c, dict) else lambda k, d=None: getattr(c, k, d)
            card_instance_id = get("card_instance_id")
            front_crop = get("front_crop")
            back_crop = get("back_crop")
            quality = get("quality", {}) or {}
            # `instance_id` is the production TEXT UUID; we reuse the
            # repository's `card_instance_id` as the value.
            if self._writer is None:
                raise RuntimeError("CardsRepository initialized without a Writer")
            self._writer.submit(Write(
                sql="""
                    INSERT OR REPLACE INTO card_instances(
                        instance_id, video_id, run_id, track_id,
                        front_crop, back_crop
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                params=(
                    card_instance_id, video_id, run_id, card_instance_id,
                    front_crop, back_crop,
                ),
            ))
            for metric, value in quality.items():
                self._writer.submit(Write(
                    sql="""
                        INSERT OR REPLACE INTO card_view_metrics(
                            card_instance_id, metric, value
                        ) VALUES (?, ?, ?)
                    """,
                    params=(card_instance_id, metric, float(value)),
                ))

    def list_for_run(self, run_id: str) -> list[dict]:
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT instance_id, front_crop, back_crop "
                "FROM card_instances WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
            out: list[dict] = []
            for instance_id, front, back in rows:
                quality = dict(conn.execute(
                    "SELECT metric, value FROM card_view_metrics "
                    "WHERE card_instance_id=?",
                    (instance_id,),
                ).fetchall())
                out.append({
                    "card_instance_id": instance_id,
                    "front_crop": front,
                    "back_crop": back,
                    "quality": quality,
                })
            return out

    def get(self, card_instance_id: str) -> dict | None:
        with read_connection(self._db_path) as conn:
            row = conn.execute(
                "SELECT instance_id, run_id, front_crop, back_crop "
                "FROM card_instances WHERE instance_id=?",
                (card_instance_id,),
            ).fetchone()
        if row is None:
            return None
        with read_connection(self._db_path) as conn:
            quality = dict(conn.execute(
                "SELECT metric, value FROM card_view_metrics WHERE card_instance_id=?",
                (card_instance_id,),
            ).fetchall())
        return {
            "card_instance_id": row[0],
            "run_id": row[1],
            "front_crop": row[2],
            "back_crop": row[3],
            "quality": quality,
        }

    def add_card_instance(
        self,
        *,
        video_id: int,
        track_id: str,
        angle: str | None,
        session_id: str | None,
        reid_embedding: bytes | None,
        run_id: str | None,
    ) -> int:
        """Insert a card_instances row and return the new row id.

        Synchronous: blocks on the Writer queue until the INSERT is
        committed so callers (the store stage) can use the returned id
        immediately for subsequent card_views inserts.
        """
        from card_capture.data.sql_queries import CARDS_ADD_INSTANCE
        from card_capture.data.writer import Write

        if self._writer is None:
            raise RuntimeError("CardsRepository initialized without a Writer")

        future = self._writer.submit_returning(Write(
            sql=CARDS_ADD_INSTANCE,
            params=(video_id, track_id, angle, session_id, reid_embedding, run_id),
        ))
        return int(future.result())  # raises if the writer errored

    def update_instance_deduplication(
        self,
        *,
        row_id: int,
        primary_hash: str,
        cross_video_parent: int | None,
        reid_embedding: bytes | None = None,
    ) -> None:
        from card_capture.data.sql_queries import CARDS_UPDATE_DEDUPLICATION
        from card_capture.data.writer import Write
        if self._writer is None:
            raise RuntimeError("CardsRepository initialized without a Writer")
        self._writer.submit(Write(
            sql=CARDS_UPDATE_DEDUPLICATION,
            params=(primary_hash, cross_video_parent, reid_embedding, row_id),
        ))

    def update_instance_fusion(self, *, row_id: int, fused_image_path: str) -> None:
        from card_capture.data.sql_queries import CARDS_UPDATE_FUSION
        from card_capture.data.writer import Write
        if self._writer is None:
            raise RuntimeError("CardsRepository initialized without a Writer")
        self._writer.submit(Write(
            sql=CARDS_UPDATE_FUSION,
            params=(fused_image_path, row_id),
        ))

    def add_card_view(
        self,
        *,
        card_instance_id: int,
        frame_index: int,
        timestamp_ms: int,
        corners: list,
        confidence: float,
        rectified_path: str,
        quality_score: dict | None,
        is_canonical: bool,
        glare_x: float | None,
        glare_y: float | None,
        sharpness: float | None,
        initial_confidence: float | None,
        glare_mask_b64: str | None = None,
        laplacian_heatmap_b64: str | None = None,
        metadata_json: dict | None = None,
    ) -> int:
        from card_capture.data.sql_queries import CARDS_ADD_VIEW
        from card_capture.data.writer import Write
        if self._writer is None:
            raise RuntimeError("CardsRepository initialized without a Writer")
        future = self._writer.submit_returning(Write(
            sql=CARDS_ADD_VIEW,
            params=(
                card_instance_id, frame_index, timestamp_ms,
                json.dumps(corners), float(confidence), rectified_path,
                json.dumps(quality_score or {}), 1 if is_canonical else 0,
                glare_x, glare_y, sharpness,
                glare_mask_b64, laplacian_heatmap_b64,
                initial_confidence,
                json.dumps(metadata_json or {}),
            ),
        ))
        return int(future.result())

    def add_saved_card(
        self,
        *,
        detection_id: int,
        image_path: str,
        final_score: float,
        video_id: int = 0,
        source_path: str = "",
        timestamp_ms: int = 0,
        score_components_json: dict | None = None,
    ) -> None:
        from card_capture.data.sql_queries import CARDS_ADD_SAVED
        from card_capture.data.writer import Write
        if self._writer is None:
            raise RuntimeError("CardsRepository initialized without a Writer")
        self._writer.submit(Write(
            sql=CARDS_ADD_SAVED,
            params=(
                detection_id, video_id, image_path, float(final_score),
                source_path, timestamp_ms,
                json.dumps(score_components_json or {}),
            ),
        ))

    def add_track_telemetry(
        self,
        *,
        video_id: int,
        instance_id: str,
        frame_index: int,
        area: float,
        aspect: float,
        cx: float,
        cy: float,
    ) -> None:
        from card_capture.data.sql_queries import CARDS_ADD_TRACK_TELEMETRY
        from card_capture.data.writer import Write
        if self._writer is None:
            raise RuntimeError("CardsRepository initialized without a Writer")
        self._writer.submit(Write(
            sql=CARDS_ADD_TRACK_TELEMETRY,
            params=(video_id, instance_id, frame_index,
                    float(area), float(aspect), float(cx), float(cy)),
        ))

    def add_pipeline_event(
        self,
        *,
        video_id: int,
        frame_index: int,
        timestamp_ms: int,
        event_type: str,
        data: dict,
    ) -> None:
        from card_capture.data.sql_queries import CARDS_ADD_PIPELINE_EVENT
        from card_capture.data.writer import Write
        if self._writer is None:
            raise RuntimeError("CardsRepository initialized without a Writer")
        self._writer.submit(Write(
            sql=CARDS_ADD_PIPELINE_EVENT,
            params=(video_id, frame_index, timestamp_ms,
                    event_type, json.dumps(data)),
        ))

    def find_embeddings_excluding_video(self, *, video_id: int) -> list[tuple[int, bytes]]:
        from card_capture.data.sql_queries import CARDS_FIND_EMBEDDINGS_EXCLUDING_VIDEO
        with read_connection(self._db_path) as conn:
            rows = conn.execute(
                CARDS_FIND_EMBEDDINGS_EXCLUDING_VIDEO, (video_id,)
            ).fetchall()
        return [(int(r[0]), bytes(r[1])) for r in rows]
