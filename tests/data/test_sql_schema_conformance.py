"""Schema conformance test for all CARDS_* SQL constants in sql_queries.py.

Tests that every INSERT/UPDATE constant in the Phase 2 back-half section of
sql_queries.py executes without OperationalError/IntegrityError against a
real production database schema (STORAGE_INIT_SCHEMA + all migrations).

We bypass the Writer thread entirely to avoid the known deadlock risk on
failed fire-and-forget submits. Each SQL constant is executed directly
against a plain sqlite3 connection built from the real schema.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from migrations.run_migrations import apply_migrations
from card_capture.data.sql_queries import STORAGE_INIT_SCHEMA


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _build_schema_db(tmp_path: Path) -> sqlite3.Connection:
    """Return an in-memory-equivalent connection with STORAGE_INIT_SCHEMA +
    all migrations applied (foreign_keys ON, WAL off for speed).

    STORAGE_INIT_SCHEMA defines tables that exist in the legacy storage layer
    (track_telemetry, performance_logs, etc.) and are NOT covered by migrations.
    We apply it first (IF NOT EXISTS guards make it idempotent), then apply all
    migration files.
    """
    db_path = tmp_path / "conformance.sqlite"
    # Pre-create the tables defined only in STORAGE_INIT_SCHEMA
    with sqlite3.connect(str(db_path)) as init_conn:
        init_conn.executescript(STORAGE_INIT_SCHEMA)
    apply_migrations(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _seed_parent_rows(conn: sqlite3.Connection) -> dict:
    """Insert the minimal parent rows needed to satisfy FK constraints.

    Returns a dict of ids: {'video_id': int, 'run_id': str, ...}
    """
    cur = conn.execute(
        "INSERT INTO videos (source_path, file_hash, duration_ms, width, height, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("/test/video.mov", "abc123", 60000, 3840, 2160, "processing"),
    )
    conn.commit()
    video_id = cur.lastrowid

    conn.execute(
        "INSERT OR IGNORE INTO pipeline_runs (run_id, video_id, status) "
        "VALUES (?, ?, 'running')",
        ("run-test-001", video_id),
    )
    conn.commit()
    return {"video_id": video_id, "run_id": "run-test-001"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def schema_conn(tmp_path):
    """A real production schema connection with one video + run row seeded."""
    conn = _build_schema_db(tmp_path)
    yield conn
    conn.close()


@pytest.fixture()
def seeded(schema_conn):
    """Returns (conn, ids) where ids has video_id and run_id."""
    ids = _seed_parent_rows(schema_conn)
    return schema_conn, ids


# ---------------------------------------------------------------------------
# Import the constants under test
# ---------------------------------------------------------------------------

from card_capture.data.sql_queries import (
    CARDS_ADD_INSTANCE,
    CARDS_UPDATE_DEDUPLICATION,
    CARDS_UPDATE_FUSION,
    CARDS_ADD_VIEW,
    CARDS_ADD_SAVED,
    CARDS_ADD_TRACK_TELEMETRY,
    CARDS_ADD_PIPELINE_EVENT,
)


# ---------------------------------------------------------------------------
# Helper: count the ? placeholders in a SQL string
# ---------------------------------------------------------------------------

def _count_params(sql: str) -> int:
    return sql.count("?")


# ---------------------------------------------------------------------------
# Tests — each must pass without OperationalError or IntegrityError
# ---------------------------------------------------------------------------


class TestCardsAddInstance:
    def test_inserts_row_no_error(self, seeded):
        conn, ids = seeded
        video_id = ids["video_id"]
        conn.execute(
            CARDS_ADD_INSTANCE,
            (video_id, "track-001", "front", "session-1", None, "run-test-001"),
        )
        conn.commit()

    def test_param_count_matches_placeholders(self):
        # 6 params: video_id, track_id, angle, session_id, reid_embedding, run_id
        assert _count_params(CARDS_ADD_INSTANCE) == 6


class TestCardsUpdateDeduplication:
    def test_updates_row_no_error(self, seeded):
        conn, ids = seeded
        video_id = ids["video_id"]
        cur = conn.execute(
            CARDS_ADD_INSTANCE,
            (video_id, "track-dedup", "front", None, None, "run-test-001"),
        )
        conn.commit()
        row_id = cur.lastrowid
        # primary_hash -> visual_hash is the bug; this should not raise
        conn.execute(
            CARDS_UPDATE_DEDUPLICATION,
            ("aabbccdd", None, None, row_id),
        )
        conn.commit()

    def test_param_count_matches_placeholders(self):
        # 4 params: visual_hash, is_duplicate_of, reid_embedding, id
        assert _count_params(CARDS_UPDATE_DEDUPLICATION) == 4


class TestCardsUpdateFusion:
    def test_updates_row_no_error(self, seeded):
        conn, ids = seeded
        video_id = ids["video_id"]
        cur = conn.execute(
            CARDS_ADD_INSTANCE,
            (video_id, "track-fusion", "front", None, None, "run-test-001"),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.execute(CARDS_UPDATE_FUSION, ("/out/fused.jpg", row_id))
        conn.commit()

    def test_param_count_matches_placeholders(self):
        assert _count_params(CARDS_UPDATE_FUSION) == 2


class TestCardsAddView:
    def _insert_instance(self, conn, video_id):
        cur = conn.execute(
            CARDS_ADD_INSTANCE,
            (video_id, "track-view", "front", None, None, "run-test-001"),
        )
        conn.commit()
        return cur.lastrowid

    def test_inserts_row_no_error(self, seeded):
        conn, ids = seeded
        inst_id = self._insert_instance(conn, ids["video_id"])
        # corners->corners_json, quality_score->quality_score_json are the bugs
        # Also metadata_json is NOT NULL — must be provided
        conn.execute(
            CARDS_ADD_VIEW,
            (
                inst_id,
                42,           # frame_index
                1000,         # timestamp_ms
                json.dumps([[0, 0], [750, 0], [750, 1050], [0, 1050]]),  # corners_json
                0.95,         # confidence
                "/out/rect.jpg",  # rectified_path
                json.dumps({"sharpness": 0.8}),  # quality_score_json
                1,            # is_canonical
                None,         # glare_x
                None,         # glare_y
                0.8,          # sharpness
                None,         # glare_mask_b64
                None,         # laplacian_heatmap_b64
                0.95,         # initial_confidence
                json.dumps({}),  # metadata_json
            ),
        )
        conn.commit()

    def test_param_count_matches_placeholders(self):
        # 15 params: card_instance_id, frame_index, timestamp_ms, corners_json,
        #            confidence, rectified_path, quality_score_json, is_canonical,
        #            glare_x, glare_y, sharpness, glare_mask_b64,
        #            laplacian_heatmap_b64, initial_confidence, metadata_json
        assert _count_params(CARDS_ADD_VIEW) == 15


class TestCardsAddSaved:
    def _insert_instance_and_view(self, conn, video_id):
        cur = conn.execute(
            CARDS_ADD_INSTANCE,
            (video_id, "track-saved", "front", None, None, "run-test-001"),
        )
        conn.commit()
        inst_id = cur.lastrowid
        cur2 = conn.execute(
            CARDS_ADD_VIEW,
            (
                inst_id, 0, 0,
                json.dumps([]),
                0.9, "/out/r.jpg",
                json.dumps({}),
                0,
                None, None, None, None, None, None,
                json.dumps({}),
            ),
        )
        conn.commit()
        return cur2.lastrowid

    def test_inserts_row_no_error(self, seeded):
        conn, ids = seeded
        view_id = self._insert_instance_and_view(conn, ids["video_id"])
        # saved_cards requires video_id, source_path, timestamp_ms,
        # score_components_json — missing those is the bug
        conn.execute(
            CARDS_ADD_SAVED,
            (view_id, ids["video_id"], "/out/card.jpg", 0.88, "/test/video.mov", 1000, json.dumps({"sharpness": 0.8})),
        )
        conn.commit()

    def test_param_count_matches_placeholders(self):
        # 7 params: detection_id, video_id, image_path, final_score,
        #           source_path, timestamp_ms, score_components_json
        assert _count_params(CARDS_ADD_SAVED) == 7


class TestCardsAddTrackTelemetry:
    def test_inserts_row_no_error(self, seeded):
        conn, ids = seeded
        video_id = ids["video_id"]
        # instance_id, area, aspect, cx, cy are the buggy column names
        conn.execute(
            CARDS_ADD_TRACK_TELEMETRY,
            (video_id, "track-001", 42, 750000.0, 0.714, 1920.0, 1080.0),
        )
        conn.commit()

    def test_param_count_matches_placeholders(self):
        # 7 params: video_id, track_id, frame_index,
        #           polygon_area, aspect_ratio, centroid_x, centroid_y
        assert _count_params(CARDS_ADD_TRACK_TELEMETRY) == 7


class TestCardsAddPipelineEvent:
    def test_inserts_row_no_error(self, seeded):
        conn, ids = seeded
        video_id = ids["video_id"]
        # data -> data_json is the bug
        conn.execute(
            CARDS_ADD_PIPELINE_EVENT,
            (video_id, 0, 0, "test_event", json.dumps({"key": "val"})),
        )
        conn.commit()

    def test_param_count_matches_placeholders(self):
        # 5 params: video_id, frame_index, timestamp_ms, event_type, data_json
        assert _count_params(CARDS_ADD_PIPELINE_EVENT) == 5
