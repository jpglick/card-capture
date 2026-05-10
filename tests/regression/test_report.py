import json
from pathlib import Path

from tests.regression.cross_video import DedupMetrics
from tests.regression.metrics import VideoMetrics
from tests.regression.report import write_json_report, write_markdown_report, AggregateReport


def _vm(video_id, recall=1.0, phantom=0.0, fb=1.0):
    return VideoMetrics(
        video_id=video_id, expected_cards=2, detected_cards=2,
        recall=recall, phantom_count=0, pipeline_output_count=2,
        phantom_rate=phantom, fb_correct=2, fb_total=2, fb_accuracy=fb,
        id_switches=0, sharpness_mean=100.0, wall_clock_s=12.5, peak_memory_mb=512.0,
    )


def test_write_json_report_round_trips(tmp_path):
    agg = AggregateReport(
        git_sha="abc123",
        per_video=(_vm("v1"), _vm("v2", recall=0.5)),
        dedup=DedupMetrics(1, 0, 0, 1.0, 1.0, 1.0),
    )
    path = tmp_path / "report.json"
    write_json_report(agg, path)

    loaded = json.loads(path.read_text())
    assert loaded["git_sha"] == "abc123"
    assert len(loaded["per_video"]) == 2
    assert loaded["per_video"][1]["recall"] == 0.5
    assert loaded["dedup"]["f1"] == 1.0


def test_write_markdown_report_includes_deltas(tmp_path):
    baseline = AggregateReport(
        git_sha="aaa", per_video=(_vm("v1", recall=0.6, phantom=0.4),),
        dedup=DedupMetrics(0, 0, 0, 1.0, 1.0, 1.0),
    )
    current = AggregateReport(
        git_sha="bbb", per_video=(_vm("v1", recall=0.9, phantom=0.1),),
        dedup=DedupMetrics(0, 0, 0, 1.0, 1.0, 1.0),
    )
    path = tmp_path / "report.md"
    write_markdown_report(current, path, baseline=baseline)

    text = path.read_text()
    assert "v1" in text
    assert "0.900" in text or "0.90" in text
    assert "+0.300" in text or "+0.30" in text
