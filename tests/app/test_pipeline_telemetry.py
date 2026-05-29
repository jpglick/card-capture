def test_event_bus_telemetry_emits_progress_event():
    from unittest.mock import MagicMock
    from app.services.pipeline_telemetry import EventBusTelemetry
    bus = MagicMock()
    tel = EventBusTelemetry(bus, "run-1")
    tel.progress("detect", 50, "batch 5/10")
    
    # Verify bus.emit was called with stage_progress
    bus.emit.assert_called()
    args = bus.emit.call_args[0]
    assert args[0] == "run-1"
    assert args[1].name == "stage_progress"
    assert args[1].payload["pct"] == 50
