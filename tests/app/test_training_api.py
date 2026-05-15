import sqlite3, tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient


def _bootstrap_app(tmp: Path):
    from app.main import create_app
    db = tmp / "cards.sqlite"
    from migrations.run_migrations import apply_migrations
    apply_migrations(db)
    app = create_app(db_path=db)
    return TestClient(app), db


def test_presence_next_empty_returns_204():
    with tempfile.TemporaryDirectory() as tmp:
        client, _ = _bootstrap_app(Path(tmp))
        r = client.get("/api/v1/training/presence/next")
        assert r.status_code == 204


def test_presence_label_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        client, db = _bootstrap_app(Path(tmp))
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "INSERT INTO presence_samples (run_id, video_id, frame_index, timestamp_ms, image_path) "
                "VALUES ('r1', 1, 0, 0, '/tmp/x.jpg')"
            )
            conn.commit()
        r = client.get("/api/v1/training/presence/next")
        assert r.status_code == 200
        sample_id = r.json()["sample_id"]

        r2 = client.post("/api/v1/training/presence/label",
                         json={"sample_id": sample_id, "label": "present"})
        assert r2.status_code == 204

        r3 = client.get("/api/v1/training/presence/next")
        assert r3.status_code == 204


def test_stats_returns_pending_counts():
    with tempfile.TemporaryDirectory() as tmp:
        client, db = _bootstrap_app(Path(tmp))
        r = client.get("/api/v1/training/stats")
        assert r.status_code == 200
        data = r.json()
        assert "pending" in data
        assert "presence" in data["pending"]
        assert "fb" in data["pending"]
        assert "corners" in data["pending"]
