"""Phase 1 — pipeline_runner merges PipelineConfig into request.config."""
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.event_bus import EventBus
from app.services.pipeline_runner import PipelineRunner


def test_unified_inprocess_passes_config_dict(tmp_path):
    captured: dict = {}

    def fake_runtime_run(self, request):
        captured["config"] = dict(request.config)
        result = MagicMock()
        result.manifest.contract_violations = []
        return result

    with patch(
        "card_capture.pipeline.runtime_local.LocalPipelineRuntime.run",
        new=fake_runtime_run,
    ):
        bus = EventBus()
        db = tmp_path / "cards.sqlite"
        db.touch()
        runner = PipelineRunner(bus=bus, flow_cls=None, db_path=db)
        runner._run_unified_inprocess(
            run_id="r1",
            video_id=42,
            video=str(tmp_path / "v.mov"),
            output_dir="out",
            db=str(db),
            detector="fake",
            config_preset="balanced",
        )

    assert captured["config"]["novelty_floor"] == 0.30
    assert captured["config"]["foil_threshold"] == 50.0
    assert captured["config"]["use_fb_classifier"] is True
    assert captured["config"]["detector"] == "fake"
