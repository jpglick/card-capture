from __future__ import annotations

from pathlib import Path

from card_capture.data.repositories.telemetry import TelemetryRepository
from card_capture.data.writer import Writer


def test_record_and_list(prod_db: Path) -> None:
    writer = Writer(prod_db); writer.start()
    try:
        repo = TelemetryRepository(writer=writer, db_path=prod_db)
        repo.record_event(run_id="r1", kind="stage_started", payload={"stage": "detect"})
        repo.record_event(run_id="r1", kind="stage_finished",
                          payload={"stage": "detect", "elapsed_ms": 9})
        writer.flush()
        events = repo.list_for_run("r1")
    finally:
        writer.stop()

    kinds = [e["kind"] for e in events]
    assert kinds == ["stage_started", "stage_finished"]
