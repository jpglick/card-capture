import json
from pathlib import Path

import pytest

from tests.regression.harness import run_corpus, HarnessConfig


def test_run_corpus_returns_aggregate_report(tmp_path, monkeypatch):
    video_id_str = "vid_001"
    corpus_dir = tmp_path / "corpus"
    video_dir = corpus_dir / video_id_str
    video_dir.mkdir(parents=True)
    truth = {
        "video_id": video_id_str,
        "video_path": str(video_dir / "fake.mp4"),
        "expected_cards": [
            {"card_id": "c1", "front_present": True, "back_present": False,
             "approx_front_window_ms": [0, 1000]},
        ],
    }
    (video_dir / f"{video_id_str}.truth.json").write_text(json.dumps(truth))

    from tests.regression import harness as harness_mod
    from tests.regression.pipeline_runner import HarnessInstance

    def fake_runner(video_path, db_path, output_dir, presence_threshold=0.5):
        return [
            HarnessInstance(
                instance_id=10, video_id=1, session_id=1, angle="Front",
                duplicate_of=None, fused_image_path=None,
                start_ms=100, end_ms=900, detection_count=5, phash=None,
            ),
        ], 1.5, 256.0, []

    monkeypatch.setattr(harness_mod, "run_pipeline_for_video", fake_runner)

    cfg = HarnessConfig(corpus_dir=corpus_dir, output_dir=tmp_path / "out", git_sha="testsha")
    report = run_corpus(cfg)

    assert report.git_sha == "testsha"
    assert len(report.per_video) == 1
    assert report.per_video[0].recall == 1.0
