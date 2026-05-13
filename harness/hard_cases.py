"""Hard-case capture persistence.

Hard cases are edge-case frames captured by the pipeline for operator review.
They are stored in the ``hard_cases`` table.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


def record_hard_case(
    *,
    db_path: Path,
    run_id: Optional[str] = None,
    frame_index: Optional[int] = None,
    stage_id: str,
    reason: str,
    thumbnail_path: Optional[str] = None,
    source_frame_path: Optional[str] = None,
) -> int:
    """Record a hard case in the database.

    Parameters
    ----------
    db_path:
        Path to ``cards.sqlite``.
    run_id:
        Optional pipeline run identifier.
    frame_index:
        Index of the frame where the hard case occurred.
    stage_id:
        Pipeline stage where the case was captured (e.g. "novelty").
    reason:
        Short string explaining why it's a hard case (e.g. "low_confidence").
    thumbnail_path:
        Path to a small thumbnail image.
    source_frame_path:
        Path to the full-resolution source frame.

    Returns
    -------
    int: The new case_id.
    """
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO hard_cases(
                run_id, frame_index, stage_id, reason, thumbnail_path, source_frame_path
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                frame_index,
                stage_id,
                reason,
                thumbnail_path,
                source_frame_path,
            ),
        )
        conn.commit()
        return cur.lastrowid
