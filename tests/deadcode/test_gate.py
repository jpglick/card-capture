from scripts.deadcode.gate import run_gate
from scripts.deadcode.models import GateResult


def _ok(stage): return GateResult(True, stage, "ok")
def _fail(stage): return GateResult(False, stage, "boom")


def test_gate_runs_stages_in_order_and_stops_on_first_failure(monkeypatch):
    calls = []
    monkeypatch.setattr("scripts.deadcode.gate.stage_tests",
                        lambda: calls.append("tests") or _ok("tests"))
    monkeypatch.setattr("scripts.deadcode.gate.stage_video_smoke",
                        lambda: calls.append("video") or _fail("video_smoke"))
    monkeypatch.setattr("scripts.deadcode.gate.stage_metric_regression",
                        lambda: calls.append("metric") or _ok("metric_regression"))
    result = run_gate()
    assert result.passed is False
    assert result.stage == "video_smoke"
    assert calls == ["tests", "video"]  # metric never ran


def test_gate_passes_when_all_stages_pass(monkeypatch):
    monkeypatch.setattr("scripts.deadcode.gate.stage_tests", lambda: _ok("tests"))
    monkeypatch.setattr("scripts.deadcode.gate.stage_video_smoke", lambda: _ok("video_smoke"))
    monkeypatch.setattr("scripts.deadcode.gate.stage_metric_regression",
                        lambda: _ok("metric_regression"))
    result = run_gate()
    assert result.passed is True
    assert result.stage == "all"
