"""TrainingRepository tests."""
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.training import TrainingRepository
from card_capture.data.writer import Write, Writer


def _seed_presence(prod_db: Path) -> int:
    writer = Writer(prod_db)
    writer.start()
    try:
        writer.submit(
            Write(
                sql="INSERT INTO videos(source_path, file_hash, duration_ms, width, height) VALUES (?, ?, ?, ?, ?)",
                params=("/tmp/v.mp4", "hash1", 1000, 1920, 1080),
            )
        )
        writer.submit(
            Write(
                sql="INSERT INTO pipeline_runs(run_id, video_id, status) VALUES (?, (SELECT id FROM videos ORDER BY id DESC LIMIT 1), ?)",
                params=("run-1", "running"),
            )
        )
        writer.submit(
            Write(
                sql="INSERT INTO presence_samples(run_id, video_id, frame_index, timestamp_ms, image_path, created_at) VALUES (?, (SELECT id FROM videos ORDER BY id DESC LIMIT 1), ?, ?, ?, datetime('now'))",
                params=("run-1", 42, 4200, "/tmp/present.png"),
            )
        )
        writer.flush()
        with prod_db.open("rb"):
            pass
    finally:
        writer.stop()

    import sqlite3

    with sqlite3.connect(str(prod_db)) as conn:
        return int(conn.execute("SELECT id FROM presence_samples ORDER BY id DESC LIMIT 1").fetchone()[0])


def test_label_presence_updates_row(prod_db: Path) -> None:
    sample_id = _seed_presence(prod_db)
    writer = Writer(prod_db)
    writer.start()
    try:
        repo = TrainingRepository(writer=writer, db_path=prod_db)
        repo.label_presence(sample_id, "present")
        writer.flush()
        sample = repo.next_presence_sample()
    finally:
        writer.stop()
    assert sample is None


def test_record_and_get_latest_model(prod_db: Path) -> None:
    writer = Writer(prod_db)
    writer.start()
    try:
        repo = TrainingRepository(writer=writer, db_path=prod_db)
        repo.record_model_version(
            name="fb",
            hash="abc123",
            metrics={"acc": 0.95},
            path="/tmp/model.pt",
        )
        writer.flush()
        row = repo.get_latest_model("fb")
    finally:
        writer.stop()
    assert row is not None
    assert row["model_name"] == "fb"
    assert row["training_set_hash"] == "abc123"
