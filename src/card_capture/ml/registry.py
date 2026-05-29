from typing import Optional
"""Model-versions registry — thin read/write wrapper for the ``model_versions`` table.

Schema (from Contract 1 / migrations/0001_v4_schema.sql):

    model_versions(
        version_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name          TEXT    NOT NULL,
        training_set_hash   TEXT    NOT NULL,
        eval_metrics_json   TEXT    NOT NULL,
        checkpoint_path     TEXT    NOT NULL,
        created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE (model_name, training_set_hash)
    )
"""
import json
from dataclasses import dataclass
from pathlib import Path

from card_capture.data.connection import open_connection, read_connection
from card_capture.data.sql_queries import ML_GET_LATEST, ML_LIST_MODELS, ML_REGISTER_INSERT
from card_capture.data.writer import Writer

@dataclass
class ModelVersion:
    version_id: int
    model_name: str
    training_set_hash: str
    eval_metrics: dict
    checkpoint_path: str
    created_at: str


def register_model(
    *,
    db_path: Path,
    model_name: str,
    training_set_hash: str,
    eval_metrics: dict,
    checkpoint_path: str,
) -> int:
    """Add a new row into ``model_versions`` and return its ``version_id``.

    Raises ``sqlite3.IntegrityError`` if the same ``(model_name,
    training_set_hash)`` pair already exists (the UNIQUE constraint prevents
    duplicate retrains on identical data).
    """
    writer = Writer(db_path)
    writer.start()
    try:
        with writer.serialize():
            with open_connection(db_path) as conn:
                cur = conn.execute(
                    ML_REGISTER_INSERT,
                    (model_name, training_set_hash, json.dumps(eval_metrics), checkpoint_path),
                )
                conn.commit()
                return int(cur.lastrowid)
    finally:
        writer.stop()


def get_latest(*, db_path: Path, model_name: str) -> Optional[ModelVersion]:
    """Return the most-recently created version for *model_name*, or ``None``."""
    with read_connection(db_path) as conn:
        row = conn.execute(ML_GET_LATEST, (model_name,)).fetchone()
    if row is None:
        return None
    return ModelVersion(row[0], row[1], row[2], json.loads(row[3]), row[4], row[5])


def list_models(*, db_path: Path) -> list[ModelVersion]:
    """Return all registered model versions, newest first."""
    with read_connection(db_path) as conn:
        rows = conn.execute(ML_LIST_MODELS).fetchall()
    return [ModelVersion(r[0], r[1], r[2], json.loads(r[3]), r[4], r[5]) for r in rows]
