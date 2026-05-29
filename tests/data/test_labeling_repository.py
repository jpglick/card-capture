from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.labeling import LabelingRepository
from card_capture.data.writer import Writer


def test_store_and_list_fb_labels(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = LabelingRepository(writer=writer, db_path=prod_db)
        repo.store_fb_label(instance_id="cardA", frame_index=10, side="front")
        repo.store_fb_label(instance_id="cardA", frame_index=22, side="back")
        writer.flush()
        rows = repo.list_for_instance("cardA")
    finally:
        writer.stop()

    assert {(r["frame_index"], r["side"]) for r in rows} == {(10, "front"), (22, "back")}


def test_store_and_get_truth_payload(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = LabelingRepository(writer=writer, db_path=prod_db)
        payload = {"cards": [{"id": 1}], "ts": 12345}
        repo.store_truth_payload(video_id="V001", payload=payload)
        writer.flush()
        got = repo.get_truth_payload("V001")
    finally:
        writer.stop()

    assert got == payload
