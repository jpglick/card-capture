"""Cards repository.

Writes to the production `card_instances` table (extended with front_crop /
back_crop columns by migration 0013) and the v5.5 `card_view_metrics` table.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

from card_capture.data.connection import read_connection
from card_capture.data.writer import Writer, Write
from card_capture.pipeline.request import CardRecord


class CardsRepository:
    def __init__(self, writer: Writer | None, db_path: Path | str) -> None:
        self._writer = writer
        self._db_path = Path(db_path)

    def store_final_cards(
        self,
        *,
        run_id: str,
        video_id: int,
        cards: Iterable[CardRecord],
    ) -> None:
        for c in cards:
            # `instance_id` is the production TEXT UUID; we reuse the
            # repository's `card_instance_id` as the value.
            self._writer.submit(Write(
                sql="""
                    INSERT OR REPLACE INTO card_instances(
                        instance_id, video_id, run_id, track_id,
                        front_crop, back_crop
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                params=(
                    c.card_instance_id, video_id, run_id, c.card_instance_id,
                    c.front_crop, c.back_crop,
                ),
            ))
            for metric, value in c.quality.items():
                self._writer.submit(Write(
                    sql="""
                        INSERT OR REPLACE INTO card_view_metrics(
                            card_instance_id, metric, value
                        ) VALUES (?, ?, ?)
                    """,
                    params=(c.card_instance_id, metric, float(value)),
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
