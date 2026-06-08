# tests/app/test_pipeline_telemetry_db.py
from pathlib import Path
from app.services.pipeline_telemetry import DbTelemetry
from card_capture.data.writer import Writer
from card_capture.data.repositories.telemetry import TelemetryRepository
from card_capture.data.connection import open_connection

def test_db_telemetry_records_events(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    open_connection(db_path).execute(
        "CREATE TABLE telemetry_events (id INTEGER PRIMARY KEY, run_id TEXT, kind TEXT, payload TEXT, at_ms INTEGER)"
    )
    writer = Writer(db_path)
    repo = TelemetryRepository(writer, db_path)
    
    writer.start()
    try:
        sink = DbTelemetry(repo, "run_1")
        sink.stage_started("detect", {"custom": 1})
        sink.stage_finished("detect", 50, {"meta": "data"})
        sink.progress("detect", 50, "halfway")
        sink.resource_sample({"cpu": 10})
        sink.contract_violation("too_slow", {"threshold": 100})
        writer.flush()
    finally:
        writer.stop()
        
    events = repo.list_for_run("run_1")
    assert len(events) == 5
    assert events[0]["kind"] == "stage_started"
    assert '"stage": "detect"' in events[0]["payload"]
    assert '"custom": 1' in events[0]["payload"]

    assert events[1]["kind"] == "stage_finished"
    assert '"elapsed_ms": 50' in events[1]["payload"]
    assert '"meta": "data"' in events[1]["payload"]

    assert events[2]["kind"] == "progress"
    assert '"pct": 50' in events[2]["payload"]
    assert '"detail": "halfway"' in events[2]["payload"]

    assert events[3]["kind"] == "resource_sample"
    assert '"cpu": 10' in events[3]["payload"]

    assert events[4]["kind"] == "contract_violation"
    assert '"code": "too_slow"' in events[4]["payload"]
