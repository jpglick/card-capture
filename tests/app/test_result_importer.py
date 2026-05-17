"""Tests for ResultImporter — uses synthetic tarballs."""
import io
import json
import sqlite3
import tarfile
import tempfile
from pathlib import Path

import pytest

from app.services.result_importer import ResultImporter


def _make_tarball(cards: list[dict], crop_filenames: list[str]) -> Path:
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
    return Path(tmp.name)


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "cards.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE card_instances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, track_id TEXT, session_id INTEGER,
            fused_image_path TEXT, angle TEXT,
            hidden INTEGER DEFAULT 0,
            UNIQUE(run_id, track_id)
        )""")
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
