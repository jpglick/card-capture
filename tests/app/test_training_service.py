"""Phase 1 — training_service threads PipelineConfig into runtime request."""
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.training_service import TrainingService


def test_to_url_maps_output_path_to_files(tmp_path):
    """Presence/corner images live under var/output and are served at /files.
    _to_url must root the URL at var/output (preserving the presence_samples
    subdir), regardless of where the DB lives. Regression for broken images in
    the labeling UI after the var/ reorg split db and output dirs."""
    svc = TrainingService(db_path=tmp_path / "var" / "db" / "cards.sqlite")
    abs_path = "/Users/x/proj/var/output/presence_samples/run_a_0.jpg"
    assert svc._to_url(abs_path) == "/files/presence_samples/run_a_0.jpg"


def test_rerun_video_passes_full_config(tmp_path, monkeypatch):
    captured = {}

    def fake_runtime_run(self, request):
        captured["config"] = dict(request.config)
        result = MagicMock()
        result.manifest.contract_violations = []
        return result

    db = tmp_path / "cards.sqlite"
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY, cards_extracted INT)")
        conn.execute("INSERT INTO pipeline_runs VALUES ('benchmark-x', 3)")

    svc = TrainingService(db_path=db)
    video = tmp_path / "v.mov"
    video.touch()

    with patch(
        "card_capture.pipeline.runtime_local.LocalPipelineRuntime.run",
        new=fake_runtime_run,
    ):
        # We don't care about the return value; only that config got threaded
        try:
            svc._rerun_video(str(video), 123)
        except Exception:
            pass  # tolerate the DB query at the end

    assert "foil_threshold" in captured["config"]
