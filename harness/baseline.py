"""Regression baseline persistence and retrieval.

Baselines are named snapshots of pipeline performance (metrics + config) stored
in the ``regression_baselines`` and ``regression_runs`` tables.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from card_capture.data.connection import open_connection, read_connection
from card_capture.data.sql_queries import (
    HARNESS_BASELINE_GET,
    HARNESS_BASELINE_ID_BY_NAME,
    HARNESS_BASELINE_INSERT,
    HARNESS_BASELINES_LIST,
    HARNESS_PRAGMA_FK_ON,
    HARNESS_RUN_INSERT,
)


@dataclass(frozen=True)
class Baseline:
    """A named performance baseline."""

    name: str
    code_sha: str
    config: dict
    metrics: dict
    per_video: list[dict]


def freeze_baseline(
    *,
    db_path: Path,
    name: str,
    code_sha: str,
    config: dict,
    metrics: dict,
    per_video: list[dict],
) -> int:
    """Freeze current results as a named baseline.

    Parameters
    ----------
    db_path:
        Path to ``cards.sqlite``.
    name:
        Unique name for the baseline (e.g. "baseline_v4.1").
    code_sha:
        Git commit hash.
    config:
        Full pipeline configuration dictionary.
    metrics:
        Aggregate metrics dictionary.
    per_video:
        List of per-video metric dictionaries.

    Returns
    -------
    int: The new baseline_id.
    """
    with open_connection(db_path) as conn:
        conn.execute(HARNESS_PRAGMA_FK_ON)
        cur = conn.execute(
            HARNESS_BASELINE_INSERT,
            (name, code_sha, json.dumps(config)),
        )
        baseline_id = cur.lastrowid

        conn.execute(
            HARNESS_RUN_INSERT,
            (
                baseline_id,
                code_sha,
                json.dumps(config),
                json.dumps(metrics),
                json.dumps(per_video),
            ),
        )
        return baseline_id


def persist_run(
    *,
    db_path: Path,
    baseline_name: str,
    code_sha: str,
    config: dict,
    metrics: dict,
    per_video: list[dict],
) -> int:
    """Persist a harness run associated with a baseline.

    Parameters
    ----------
    db_path:
        Path to ``cards.sqlite``.
    baseline_name:
        Name of the baseline this run is compared against.
    code_sha:
        Git commit hash.
    config:
        Full pipeline configuration dictionary.
    metrics:
        Aggregate metrics dictionary.
    per_video:
        List of per-video metric dictionaries.

    Returns
    -------
    int: The new run_id.
    """
    with open_connection(db_path) as conn:
        conn.execute(HARNESS_PRAGMA_FK_ON)
        row = conn.execute(
            HARNESS_BASELINE_ID_BY_NAME,
            (baseline_name,),
        ).fetchone()
        if not row:
            raise ValueError(f"Baseline '{baseline_name}' not found.")
        baseline_id = row[0]

        cur = conn.execute(
            HARNESS_RUN_INSERT,
            (
                baseline_id,
                code_sha,
                json.dumps(config),
                json.dumps(metrics),
                json.dumps(per_video),
            ),
        )
        return cur.lastrowid


def get_baseline(*, db_path: Path, name: str) -> Baseline:
    """Retrieve a baseline by name."""
    with read_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(HARNESS_BASELINE_GET, (name,)).fetchone()

        if not row:
            raise ValueError(f"Baseline '{name}' not found.")

        return Baseline(
            name=row["name"],
            code_sha=row["code_sha"],
            config=json.loads(row["config_json"]),
            metrics=json.loads(row["metrics_json"]),
            per_video=json.loads(row["per_video_json"]),
        )


def list_baselines(*, db_path: Path) -> list[dict[str, Any]]:
    """Return a list of all baselines with metadata."""
    with read_connection(db_path) as conn:
        rows = conn.execute(HARNESS_BASELINES_LIST).fetchall()
        keys = ("baseline_id", "name", "code_sha", "created_at")
        return [dict(zip(keys, r)) for r in rows]
