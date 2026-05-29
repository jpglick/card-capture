"""MLRepository tests."""
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.ml import MLRepository
from card_capture.data.writer import Write, Writer


def test_list_presence_training_samples(prod_db: Path) -> None:
    writer = Writer(prod_db)
    writer.start()
    try:
        writer.submit(
            Write(
                sql="INSERT INTO videos(source_path, file_hash, duration_ms, width, height) VALUES (?, ?, ?, ?, ?)",
                params=("/tmp/v.mp4", "hash-ml", 1000, 1920, 1080),
            )
        )
        writer.submit(
            Write(
                sql="INSERT INTO pipeline_runs(run_id, video_id, status) VALUES (?, (SELECT id FROM videos ORDER BY id DESC LIMIT 1), ?)",
                params=("run-ml", "running"),
            )
        )
        writer.submit(
            Write(
                sql="INSERT INTO presence_samples(run_id, video_id, frame_index, timestamp_ms, image_path, label, created_at, labeled_at) VALUES (?, (SELECT id FROM videos ORDER BY id DESC LIMIT 1), ?, ?, ?, ?, datetime('now'), datetime('now'))",
                params=("run-ml", 1, 100, "/tmp/p1.png", "present"),
            )
        )
        writer.flush()
    finally:
        writer.stop()

    repo = MLRepository(writer=None, db_path=prod_db)
    rows = repo.list_presence_training_samples()
    assert any(r["image_path"] == "/tmp/p1.png" and r["label"] == "present" for r in rows)
