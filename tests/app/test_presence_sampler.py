import sqlite3, tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest
from app.services.presence_sampler import sample_presence_frames, SAMPLES_PER_RUN


def _make_fake_video(path: Path, n_frames: int = 120, fps: float = 30.0):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))
    for i in range(n_frames):
        frame = np.full((480, 640, 3), i % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def _make_db(path: Path):
    with sqlite3.connect(str(path)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS presence_samples (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                video_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                label TEXT,
                labeled_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)


def test_sample_presence_frames_inserts_rows():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video = tmp / "test.avi"
        db = tmp / "cards.sqlite"
        _make_fake_video(video)
        _make_db(db)

        n = sample_presence_frames(
            video_path=video,
            run_id="run_test",
            video_id=1,
            output_dir=tmp,
            db_path=db,
        )

        assert n == SAMPLES_PER_RUN
        with sqlite3.connect(str(db)) as conn:
            rows = conn.execute("SELECT * FROM presence_samples").fetchall()
        assert len(rows) == SAMPLES_PER_RUN


def test_sample_saves_192px_jpegs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video = tmp / "test.avi"
        db = tmp / "cards.sqlite"
        _make_fake_video(video)
        _make_db(db)
        sample_presence_frames(video, "run_test", 1, tmp, db)
        jpegs = list((tmp / "presence_samples").glob("*.jpg"))
        assert len(jpegs) == SAMPLES_PER_RUN
        img = cv2.imread(str(jpegs[0]))
        assert img.shape[1] == 192
