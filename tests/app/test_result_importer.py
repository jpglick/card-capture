"""Tests for ResultImporter — uses synthetic tarballs."""
import io
import json
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import pytest

from app.services.result_importer import ResultImporter


def _make_tarball(cards: list[dict], crop_filenames: list[str], worker_db: Optional[Path] = None) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    with tarfile.open(tmp.name, "w:gz") as tar:
        # Add fake crop files
        for fname in crop_filenames:
            data = b"JPEG"
            info = tarfile.TarInfo(name=f"crops/{fname}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        # Add export.json
        export_data = json.dumps(cards).encode()
        info = tarfile.TarInfo(name="export.json")
        info.size = len(export_data)
        tar.addfile(info, io.BytesIO(export_data))
        if worker_db is not None:
            tar.add(worker_db, arcname="cards.sqlite")
    return Path(tmp.name)


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "cards.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE videos (id INTEGER PRIMARY KEY, source_path TEXT)")
        conn.execute("INSERT INTO videos (id, source_path) VALUES (7, 'video.mov')")
        conn.execute(
            "CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY, video_id INTEGER NOT NULL, "
            "status TEXT, cards_extracted INTEGER DEFAULT 0, detect_telemetry_json TEXT, host_info_json TEXT)"
        )
        conn.execute("INSERT INTO pipeline_runs (run_id, video_id, status) VALUES ('run-1', 7, 'running')")
        conn.execute("INSERT INTO pipeline_runs (run_id, video_id, status) VALUES ('run-2', 7, 'running')")
        conn.execute("INSERT INTO pipeline_runs (run_id, video_id, status) VALUES ('run-3', 7, 'running')")
        conn.execute("INSERT INTO pipeline_runs (run_id, video_id, status) VALUES ('run-worker', 7, 'running')")
        conn.execute("INSERT INTO pipeline_runs (run_id, video_id, status) VALUES ('run-handler', 7, 'running')")
        conn.execute("""CREATE TABLE card_instances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL, run_id TEXT, track_id TEXT, session_id INTEGER,
            visual_hash TEXT, reid_embedding BLOB, is_duplicate_of INTEGER,
            fused_image_path TEXT, angle TEXT,
            hidden INTEGER DEFAULT 0,
            UNIQUE(run_id, track_id)
        )""")
        conn.execute("""CREATE TABLE card_views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_instance_id INTEGER NOT NULL,
            frame_index INTEGER NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            corners_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            rectified_path TEXT,
            quality_score_json TEXT,
            is_canonical INTEGER NOT NULL DEFAULT 0,
            glare_x REAL,
            glare_y REAL,
            sharpness REAL,
            glare_mask_b64 TEXT,
            laplacian_heatmap_b64 TEXT,
            initial_confidence REAL,
            metadata_json TEXT NOT NULL,
            created_at TEXT
        )""")
        conn.execute("""CREATE TABLE pipeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            run_id TEXT,
            stage_id TEXT,
            frame_index INTEGER NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            data_json TEXT,
            artifact_ref TEXT,
            created_at TEXT
        )""")
        conn.execute("""CREATE TABLE run_resource_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            elapsed_s REAL NOT NULL,
            cpu_pct REAL,
            mem_used_mb REAL,
            mem_pct REAL,
            gpu_pct REAL,
            vram_used_mb REAL,
            decoder_pct REAL,
            encoder_pct REAL,
            mem_io_pct REAL,
            stage TEXT DEFAULT 'init'
        )""")
        conn.execute("CREATE TABLE pipeline_run_logs (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, line TEXT NOT NULL, logged_at TEXT)")
    return db


def test_import_cards_are_inserted(tmp_path):
    cards = [
        {"track_id": "abc", "session_id": 0, "fused_image_path": "crops/instance_abc_fused.jpg", "side": "Front"},
        {"track_id": "def", "session_id": 1, "fused_image_path": "crops/instance_def_fused.jpg", "side": "Back"},
    ]
    tarball = _make_tarball(cards, ["instance_abc_fused.jpg", "instance_def_fused.jpg"])
    db = _make_db(tmp_path)
    importer = ResultImporter(db_path=db, output_base=tmp_path)

    count = importer.import_tarball(tarball, "run-1")

    assert count == 2
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT track_id FROM card_instances WHERE run_id='run-1'").fetchall()
    assert {r[0] for r in rows} == {"abc", "def"}


def test_crops_are_copied(tmp_path):
    cards = [{"track_id": "x", "session_id": 0, "fused_image_path": "crops/instance_x_fused.jpg", "side": "Front"}]
    tarball = _make_tarball(cards, ["instance_x_fused.jpg"])
    db = _make_db(tmp_path)
    importer = ResultImporter(db_path=db, output_base=tmp_path)

    importer.import_tarball(tarball, "run-2")

    assert (tmp_path / "run-2" / "crops" / "instance_x_fused.jpg").exists()


def test_duplicate_import_is_idempotent(tmp_path):
    cards = [{"track_id": "dup", "session_id": 0, "fused_image_path": "crops/x.jpg", "side": "Front"}]
    tarball = _make_tarball(cards, ["x.jpg"])
    db = _make_db(tmp_path)
    importer = ResultImporter(db_path=db, output_base=tmp_path)
    importer.import_tarball(tarball, "run-3")
    importer.import_tarball(tarball, "run-3")  # second call — idempotent

    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM card_instances WHERE run_id='run-3'").fetchone()[0]
    assert count == 1


def test_import_cards_uses_local_run_video_id(tmp_path):
    cards = [{"track_id": "abc", "session_id": 0, "fused_image_path": "crops/instance_abc_fused.jpg", "side": "Front"}]
    tarball = _make_tarball(cards, ["instance_abc_fused.jpg"])
    db = _make_db(tmp_path)
    importer = ResultImporter(db_path=db, output_base=tmp_path)

    importer.import_tarball(tarball, "run-1")

    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT video_id, track_id FROM card_instances WHERE run_id='run-1'").fetchone()
    assert row == (7, "abc")


def test_import_embedded_worker_database_merges_views_events_and_samples(tmp_path):
    worker_db = tmp_path / "worker.sqlite"
    with sqlite3.connect(worker_db) as conn:
        conn.execute("""CREATE TABLE card_instances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            run_id TEXT,
            track_id TEXT NOT NULL,
            session_id TEXT,
            visual_hash TEXT,
            reid_embedding BLOB,
            is_duplicate_of INTEGER,
            angle TEXT,
            fused_image_path TEXT,
            created_at TEXT,
            updated_at TEXT
        )""")
        conn.execute("""CREATE TABLE card_views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_instance_id INTEGER NOT NULL,
            frame_index INTEGER NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            corners_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            rectified_path TEXT,
            quality_score_json TEXT,
            is_canonical INTEGER NOT NULL DEFAULT 0,
            glare_x REAL,
            glare_y REAL,
            sharpness REAL,
            glare_mask_b64 TEXT,
            laplacian_heatmap_b64 TEXT,
            initial_confidence REAL,
            metadata_json TEXT NOT NULL,
            created_at TEXT
        )""")
        conn.execute("""CREATE TABLE pipeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            run_id TEXT,
            stage_id TEXT,
            frame_index INTEGER NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            data_json TEXT,
            artifact_ref TEXT,
            created_at TEXT
        )""")
        conn.execute("""CREATE TABLE run_resource_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            elapsed_s REAL NOT NULL,
            cpu_pct REAL,
            mem_used_mb REAL,
            mem_pct REAL,
            gpu_pct REAL,
            vram_used_mb REAL,
            decoder_pct REAL,
            encoder_pct REAL,
            mem_io_pct REAL,
            stage TEXT DEFAULT 'init'
        )""")
        conn.execute("CREATE TABLE pipeline_run_logs (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, line TEXT NOT NULL, logged_at TEXT)")
        cur = conn.execute(
            "INSERT INTO card_instances (video_id, run_id, track_id, session_id, visual_hash, angle, fused_image_path) "
            "VALUES (1, 'run-worker', 'abc', '0', 'hash', 'Front', '/tmp/cc_output/run-worker/crops/instance_abc_fused.jpg')"
        )
        worker_card_id = cur.lastrowid
        conn.execute(
            "INSERT INTO card_views (card_instance_id, frame_index, timestamp_ms, corners_json, confidence, rectified_path, metadata_json) "
            "VALUES (?, 12, 400, '[]', 0.9, '/tmp/cc_output/run-worker/crops/track_abc_det_12_rectified.jpg', '{}')",
            (worker_card_id,),
        )
        conn.execute(
            "INSERT INTO pipeline_events (video_id, run_id, stage_id, frame_index, timestamp_ms, event_type, data_json) "
            "VALUES (1, 'run-worker', 'detect', 0, 0, 'stage_detect', '{\"elapsed_ms\": 42}')"
        )
        conn.execute(
            "INSERT INTO run_resource_samples "
            "(run_id, elapsed_s, cpu_pct, mem_used_mb, mem_pct, gpu_pct, vram_used_mb, decoder_pct, encoder_pct, mem_io_pct, stage) "
            "VALUES ('run-worker', 1.5, 10, 20, 30, 40, 50, 60, 70, 80, 'detect')"
        )
        conn.execute("INSERT INTO pipeline_run_logs (run_id, line) VALUES ('run-worker', '[mf] line')")

    tarball = _make_tarball(
        [],
        ["instance_abc_fused.jpg", "track_abc_det_12_rectified.jpg"],
        worker_db=worker_db,
    )
    db = _make_db(tmp_path)
    importer = ResultImporter(db_path=db, output_base=tmp_path)

    count = importer.import_tarball(tarball, "run-worker")

    assert count == 1
    with sqlite3.connect(db) as conn:
        card = conn.execute(
            "SELECT id, video_id, fused_image_path FROM card_instances WHERE run_id='run-worker'"
        ).fetchone()
        assert card[1] == 7
        assert card[2] == str(tmp_path / "run-worker" / "crops" / "instance_abc_fused.jpg")
        view = conn.execute(
            "SELECT rectified_path FROM card_views WHERE card_instance_id=?", (card[0],)
        ).fetchone()
        assert view[0] == str(tmp_path / "run-worker" / "crops" / "track_abc_det_12_rectified.jpg")
        event_video_id = conn.execute("SELECT video_id FROM pipeline_events WHERE run_id='run-worker'").fetchone()[0]
        sample = conn.execute(
            "SELECT stage, decoder_pct, encoder_pct, mem_io_pct FROM run_resource_samples WHERE run_id='run-worker'"
        ).fetchone()
        log_line = conn.execute("SELECT line FROM pipeline_run_logs WHERE run_id='run-worker'").fetchone()[0]
    assert event_video_id == 7
    assert sample == ("detect", 60.0, 70.0, 80.0)
    assert log_line == "[mf] line"


