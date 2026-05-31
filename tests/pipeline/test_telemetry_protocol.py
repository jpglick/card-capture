from card_capture.pipeline.telemetry import PipelineTelemetry

def test_telemetry_has_progress_method():
    # This will fail if PipelineTelemetry doesn't define progress()
    assert hasattr(PipelineTelemetry, "progress")
