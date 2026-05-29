"""BatchRepository tests."""
from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.batch import BatchRepository
from card_capture.data.writer import Writer


def test_enqueue_then_list_pending(prod_db: Path) -> None:
    writer = Writer(prod_db)
    writer.start()
    try:
        repo = BatchRepository(writer=writer, db_path=prod_db)
        repo.enqueue(batch_id="b1", total=3)
        repo.enqueue(batch_id="b2", total=5)
        writer.flush()
        pending = repo.list_pending()
    finally:
        writer.stop()
    by_id = {p["batch_id"]: p for p in pending}
    assert by_id["b1"]["total"] == 3
    assert by_id["b2"]["total"] == 5
    assert by_id["b1"]["status"] == "queued"


def test_update_progress_and_remove_from_pending(prod_db: Path) -> None:
    writer = Writer(prod_db)
    writer.start()
    try:
        repo = BatchRepository(writer=writer, db_path=prod_db)
        repo.enqueue(batch_id="b1", total=3)
        repo.update_progress(batch_id="b1", completed=3, failed=0, status="done")
        writer.flush()
        rows = repo.list_pending()
    finally:
        writer.stop()
    assert all(r["batch_id"] != "b1" for r in rows)

