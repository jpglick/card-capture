import sqlite3
from pathlib import Path
import pytest

from migrations.run_migrations import apply_migrations

EXPECTED_TABLES = {
    "truth_files",
    "regression_baselines",
    "regression_runs",
    "fb_labels",
    "dedup_clusters",
    "model_versions",
    "hard_cases",
}


def test_v4_schema_creates_expected_tables(tmp_path: Path):
    db_path = tmp_path / "cards.sqlite"
    sqlite3.connect(db_path).close()  # empty db
    apply_migrations(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {r[0] for r in rows}
    assert EXPECTED_TABLES.issubset(table_names)


def test_pipeline_events_has_v4_columns(tmp_path: Path):
    db_path = tmp_path / "cards.sqlite"
    # seed with the existing pipeline_events table (matches current schema)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE pipeline_events (id INTEGER PRIMARY KEY, event_type TEXT, payload TEXT)"
        )
    apply_migrations(db_path)
    with sqlite3.connect(db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(pipeline_events)").fetchall()}
    assert "stage_id" in cols
    assert "artifact_ref" in cols


def test_run_resource_samples_has_gpu_detail_columns(tmp_path: Path):
    db_path = tmp_path / "cards.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE run_resource_samples ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "run_id TEXT NOT NULL, elapsed_s REAL NOT NULL, "
            "cpu_pct REAL, mem_used_mb REAL, mem_pct REAL, "
            "gpu_pct REAL, vram_used_mb REAL, stage TEXT DEFAULT 'init')"
        )

    apply_migrations(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(run_resource_samples)").fetchall()}
    assert {"decoder_pct", "encoder_pct", "mem_io_pct"}.issubset(cols)


@pytest.mark.quarantine
def test_migrations_are_idempotent(tmp_path: Path):
    db_path = tmp_path / "cards.sqlite"
    sqlite3.connect(db_path).close()
    apply_migrations(db_path)
    apply_migrations(db_path)  # must not raise
    # Verify _migrations tracking table has exactly one row per SQL file.
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT filename FROM _migrations ORDER BY filename").fetchall()
    filenames = [r[0] for r in rows]
    assert filenames
    assert len(filenames) == len(set(filenames))
    # Verify the new tables are still present after double application.
    with sqlite3.connect(db_path) as conn:
        table_rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {r[0] for r in table_rows}
    assert EXPECTED_TABLES.issubset(table_names)
