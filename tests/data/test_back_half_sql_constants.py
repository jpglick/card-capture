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
    # Create minimum tables — column names must match the production schema
    # so that INSERT/UPDATE statements parse correctly.
    conn.executescript("""
        CREATE TABLE card_instances (
            id INTEGER PRIMARY KEY, video_id INTEGER, track_id TEXT,
            angle TEXT, session_id TEXT, reid_embedding BLOB, run_id TEXT,
            visual_hash TEXT, is_duplicate_of INTEGER, fused_image_path TEXT
        );
        CREATE TABLE card_views (
            id INTEGER PRIMARY KEY, card_instance_id INTEGER, frame_index INTEGER,
            timestamp_ms INTEGER, corners_json TEXT, confidence REAL,
            rectified_path TEXT, quality_score_json TEXT, is_canonical INTEGER,
            glare_x REAL, glare_y REAL, sharpness REAL,
            glare_mask_b64 TEXT, laplacian_heatmap_b64 TEXT,
            initial_confidence REAL, metadata_json TEXT
        );
        CREATE TABLE saved_cards (
            id INTEGER PRIMARY KEY, detection_id INTEGER, video_id INTEGER,
            image_path TEXT, final_score REAL,
            source_path TEXT, timestamp_ms INTEGER, score_components_json TEXT
        );
        CREATE TABLE track_telemetry (
            video_id INTEGER, track_id TEXT, frame_index INTEGER,
            polygon_area REAL, aspect_ratio REAL, centroid_x REAL, centroid_y REAL
        );
        CREATE TABLE pipeline_events (
            video_id INTEGER, frame_index INTEGER, timestamp_ms INTEGER,
            event_type TEXT, data_json TEXT
        );
    """)
    sql = getattr(q, name)
    try:
        conn.execute(f"EXPLAIN {sql}", tuple([None] * sql.count("?")))
    except sqlite3.OperationalError as e:
        pytest.fail(f"{name} does not parse: {e}\nSQL: {sql}")
    finally:
        conn.close()
