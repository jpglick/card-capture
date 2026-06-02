from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class HarnessInstance:
    """A pipeline-produced Card Instance, normalized for harness consumption."""
    instance_id: int
    video_id: int
    session_id: int
    angle: str
    duplicate_of: Optional[int]
    fused_image_path: Optional[str]
    start_ms: int
    end_ms: int
    detection_count: int
    phash: Optional[str]


def instances_from_db_rows(rows: Iterable[dict]) -> List[HarnessInstance]:
    out: List[HarnessInstance] = []
    for row in rows:
        out.append(
            HarnessInstance(
                instance_id=int(row["instance_id"]),
                video_id=int(row["video_id"]),
                session_id=int(row["session_id"]),
                angle=str(row.get("angle") or "Unknown"),
                duplicate_of=row.get("is_duplicate_of"),
                fused_image_path=row.get("fused_image_path"),
                start_ms=int(row["start_time"]),
                end_ms=int(row["end_time"]),
                detection_count=int(row.get("detection_count") or 0),
                phash=row.get("phash"),
            )
        )
    return out


def load_instances_for_video(db_path: Path, video_id: int) -> List[HarnessInstance]:
    """Read Card Instances for a single video out of the pipeline's SQLite DB."""
    from card_capture.stages.store.storage import Storage
    storage = Storage(db_path)
    storage.initialize()
    with storage._connect() as conn:
        # Verify whether phash column exists on card_instances
        cols = {row[1] for row in conn.execute("PRAGMA table_info(card_instances)").fetchall()}
        phash_sel = ", ci.phash" if "phash" in cols else ", NULL AS phash"
        rows = conn.execute(
            f"""
            SELECT ci.id AS instance_id, ci.video_id, ci.session_id, ci.angle,
                   ci.is_duplicate_of, ci.fused_image_path{phash_sel},
                   MIN(cv.timestamp_ms) AS start_time,
                   MAX(cv.timestamp_ms) AS end_time,
                   COUNT(cv.id) AS detection_count
            FROM card_instances ci
            LEFT JOIN card_views cv ON cv.card_instance_id = ci.id
            WHERE ci.video_id = ?
            GROUP BY ci.id
            ORDER BY start_time ASC
            """,
            (video_id,),
        ).fetchall()
    return instances_from_db_rows([dict(r) for r in rows])
