from pathlib import Path
from card_capture.data.connection import open_connection
from card_capture.data.writer import Writer
from card_capture.data.repositories.labeling import LabelingRepository


def _init_schema(db: Path):
    conn = open_connection(db)
    conn.execute("""
        CREATE TABLE fb_labels (
            label_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            source_run_id INTEGER,
            instance_id   TEXT    NOT NULL,
            frame_index   INTEGER NOT NULL,
            side          TEXT    NOT NULL CHECK (side IN ('front', 'back', 'uncertain', 'no_card')),
            labeler       TEXT,
            created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE truth_files (
            video_id        TEXT    PRIMARY KEY,
            schema_version  INTEGER NOT NULL,
            payload_json    TEXT    NOT NULL,
            updated_at      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.close()


def test_store_and_query_fb_label(tmp_path):
    db = tmp_path / "lab.db"
    _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = LabelingRepository(writer=writer, db_path=db)
        repo.store_fb_label(instance_id="inst_1", frame_index=42, label="front", labeler="human")
        writer.flush()
        labels = repo.list_for_instance("inst_1")
    finally:
        writer.stop()
    assert any(l["side"] == "front" and l["frame_index"] == 42 for l in labels)


def test_store_and_get_truth_payload(tmp_path):
    db = tmp_path / "lab2.db"
    _init_schema(db)
    writer = Writer(db); writer.start()
    try:
        repo = LabelingRepository(writer=writer, db_path=db)
        repo.store_truth_payload("vid_1", {"cards": []})
        writer.flush()
        payload = repo.get_truth_payload("vid_1")
    finally:
        writer.stop()
    assert payload == {"cards": []}
