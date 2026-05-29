"""Phase 2 — SQL constants exist and parse as valid SQL statements."""
import sqlite3
import pytest

from card_capture.data import sql_queries as q


CONSTANTS = [
    "CARDS_ADD_INSTANCE",
    "CARDS_UPDATE_DEDUPLICATION",
    "CARDS_UPDATE_FUSION",
    "CARDS_ADD_VIEW",
    "CARDS_ADD_SAVED",
    "CARDS_ADD_TRACK_TELEMETRY",
    "CARDS_ADD_PIPELINE_EVENT",
    "CARDS_FIND_EMBEDDINGS_EXCLUDING_VIDEO",
]


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_exists(name):
    assert hasattr(q, name), f"{name} missing from sql_queries"


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_is_string(name):
    val = getattr(q, name)
    assert isinstance(val, str) and len(val) > 0


@pytest.mark.parametrize("name", CONSTANTS)
def test_constant_parses(name):
    """SQLite must accept the statement for prepare (uses a scratch in-memory db)."""
    conn = sqlite3.connect(":memory:")
    # Create minimum tables for prepare to succeed
    conn.executescript("""
        CREATE TABLE card_instances (
            id INTEGER PRIMARY KEY, video_id INTEGER, track_id TEXT,
            angle TEXT, session_id TEXT, reid_embedding BLOB, run_id TEXT,
            primary_hash TEXT, is_duplicate_of INTEGER, fused_image_path TEXT
        );
        CREATE TABLE card_views (
            id INTEGER PRIMARY KEY, card_instance_id INTEGER, frame_index INTEGER,
            timestamp_ms INTEGER, corners TEXT, confidence REAL,
            rectified_path TEXT, quality_score TEXT, is_canonical INTEGER,
            glare_x REAL, glare_y REAL, sharpness REAL, initial_confidence REAL
        );
        CREATE TABLE saved_cards (
            id INTEGER PRIMARY KEY, detection_id INTEGER,
            image_path TEXT, final_score REAL
        );
        CREATE TABLE track_telemetry (
            video_id INTEGER, instance_id TEXT, frame_index INTEGER,
            area REAL, aspect REAL, cx REAL, cy REAL
        );
        CREATE TABLE pipeline_events (
            video_id INTEGER, frame_index INTEGER, timestamp_ms INTEGER,
            event_type TEXT, data TEXT
        );
    """)
    sql = getattr(q, name)
    try:
        conn.execute(f"EXPLAIN {sql}", tuple([None] * sql.count("?")))
    except sqlite3.OperationalError as e:
        pytest.fail(f"{name} does not parse: {e}\nSQL: {sql}")
    finally:
        conn.close()
