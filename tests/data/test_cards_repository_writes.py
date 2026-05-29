"""Phase 2 — CardsRepository write methods used by the store stage."""
import sqlite3
from pathlib import Path

import pytest

from card_capture.data.connection import open_connection
from card_capture.data.repositories.cards import CardsRepository
from card_capture.data.writer import Writer


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "cards.sqlite"
    with open_connection(p) as conn:
        conn.executescript("""
            CREATE TABLE card_instances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                angle TEXT,
                session_id TEXT,
                reid_embedding BLOB,
                run_id TEXT,
                primary_hash TEXT,
                is_duplicate_of INTEGER,
                fused_image_path TEXT
            );
            CREATE TABLE card_views (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_instance_id INTEGER NOT NULL,
                frame_index INTEGER,
                timestamp_ms INTEGER,
                corners TEXT,
                confidence REAL,
                rectified_path TEXT,
                quality_score TEXT,
                is_canonical INTEGER,
                glare_x REAL,
                glare_y REAL,
                sharpness REAL,
                initial_confidence REAL
            );
            CREATE TABLE saved_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER,
                image_path TEXT,
                final_score REAL
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
    return p


@pytest.fixture
def repo(db):
    from card_capture.data.writer import Writer
    w = Writer(db)
    w.start()
    yield CardsRepository(w, db)
    w.stop()


def test_add_card_instance_returns_row_id(repo, db):
    row_id = repo.add_card_instance(
        video_id=1, track_id="t-abc", angle="Front",
        session_id="0", reid_embedding=None, run_id="r1",
    )
    assert isinstance(row_id, int) and row_id > 0
    with open_connection(db) as conn:
        row = conn.execute(
            "SELECT video_id, track_id, angle, run_id FROM card_instances WHERE id=?",
            (row_id,),
        ).fetchone()
    assert tuple(row) == (1, "t-abc", "Front", "r1")


def test_update_instance_deduplication(repo, db):
    row_id = repo.add_card_instance(
        video_id=1, track_id="t1", angle="Front",
        session_id="0", reid_embedding=None, run_id="r1",
    )
    repo.update_instance_deduplication(
        row_id=row_id,
        primary_hash="aabbccdd",
        cross_video_parent=None,
        reid_embedding=b"\x00" * 4,
    )
    repo._writer.flush()
    with open_connection(db) as conn:
        row = conn.execute(
            "SELECT primary_hash, is_duplicate_of, reid_embedding FROM card_instances WHERE id=?",
            (row_id,),
        ).fetchone()
    assert row[0] == "aabbccdd"
    assert row[1] is None
    assert row[2] == b"\x00" * 4


def test_update_instance_fusion(repo, db):
    row_id = repo.add_card_instance(
        video_id=1, track_id="t1", angle="Front",
        session_id="0", reid_embedding=None, run_id="r1",
    )
    repo.update_instance_fusion(row_id=row_id, fused_image_path="/tmp/x.jpg")
    repo._writer.flush()
    with open_connection(db) as conn:
        path = conn.execute(
            "SELECT fused_image_path FROM card_instances WHERE id=?",
            (row_id,),
        ).fetchone()[0]
    assert path == "/tmp/x.jpg"


def test_add_card_view(repo, db):
    inst = repo.add_card_instance(
        video_id=1, track_id="t1", angle="Front",
        session_id="0", reid_embedding=None, run_id="r1",
    )
    import json
    view_id = repo.add_card_view(
        card_instance_id=inst,
        frame_index=10,
        timestamp_ms=333,
        corners=[(0.0, 0.0), (750.0, 0.0), (750.0, 1050.0), (0.0, 1050.0)],
        confidence=0.92,
        rectified_path="/tmp/v.jpg",
        quality_score={"sharpness": 0.7},
        is_canonical=True,
        glare_x=None, glare_y=None, sharpness=0.7,
        initial_confidence=0.92,
    )
    assert isinstance(view_id, int) and view_id > 0
    with open_connection(db) as conn:
        row = conn.execute(
            "SELECT frame_index, rectified_path, is_canonical, quality_score "
            "FROM card_views WHERE id=?",
            (view_id,),
        ).fetchone()
    assert row[0] == 10
    assert row[1] == "/tmp/v.jpg"
    assert row[2] == 1
    assert json.loads(row[3])["sharpness"] == 0.7


def test_add_saved_card(repo, db):
    repo.add_saved_card(detection_id=42, image_path="/tmp/c.jpg", final_score=0.85)
    repo._writer.flush()
    with open_connection(db) as conn:
        row = conn.execute(
            "SELECT detection_id, image_path, final_score FROM saved_cards"
        ).fetchone()
    assert tuple(row) == (42, "/tmp/c.jpg", 0.85)


def test_add_track_telemetry(repo, db):
    repo.add_track_telemetry(
        video_id=1, instance_id="t-abc", frame_index=100,
        area=750000.0, aspect=0.714, cx=1920.0, cy=1080.0,
    )
    repo._writer.flush()
    with open_connection(db) as conn:
        row = conn.execute("SELECT video_id, instance_id, frame_index, area FROM track_telemetry").fetchone()
    assert row[0] == 1
    assert row[1] == "t-abc"
    assert row[2] == 100
    assert row[3] == 750000.0


def test_add_pipeline_event(repo, db):
    import json
    repo.add_pipeline_event(
        video_id=1, frame_index=0, timestamp_ms=0,
        event_type="reid_embedding_failed",
        data={"instance_id": "t-abc", "error": "FileNotFoundError"},
    )
    repo._writer.flush()
    with open_connection(db) as conn:
        row = conn.execute("SELECT event_type, data FROM pipeline_events").fetchone()
    assert row[0] == "reid_embedding_failed"
    assert json.loads(row[1])["error"] == "FileNotFoundError"


def test_find_embeddings_excluding_video(repo, db):
    import numpy as np
    emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32).tobytes()
    emb_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32).tobytes()
    repo.add_card_instance(
        video_id=1, track_id="t1", angle="Front",
        session_id="0", reid_embedding=emb_a, run_id="r1",
    )
    iid2 = repo.add_card_instance(
        video_id=2, track_id="t2", angle="Front",
        session_id="0", reid_embedding=emb_b, run_id="r2",
    )
    repo._writer.flush()
    rows = repo.find_embeddings_excluding_video(video_id=1)
    assert len(rows) == 1
    assert rows[0][0] == iid2
    assert rows[0][1] == emb_b
