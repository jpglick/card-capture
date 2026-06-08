"""LocalPipelineRuntime forwards per-stage metrics into stage_finished.

The stub stage populates state["stage_metrics"] directly (the same dict shape
emit_stage_metrics produces) so this isolates the runtime's drain behavior
without a real DB write. The emit_stage_metrics -> buffer path is covered by
tests/pipeline/test_stage_metrics_helper.py.
"""
from __future__ import annotations

from types import SimpleNamespace

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import InMemoryTelemetry


def test_runtime_passes_stage_metrics_to_stage_finished(tmp_path):
    import card_capture.pipeline.runtime_local as rtl

    def stub_run(state, **kwargs):
        state.setdefault("stage_metrics", {})["detect"] = {"detections": 9}

    telem = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telem)
    request = PipelineRunRequest(
        run_id="r1",
        input_video="artifact://local/v.mov",
        output_root=f"artifact://local/{tmp_path}/out/",
        runtime_mode="cpu_debug",
        config={},
        db_path=str(tmp_path / "cards.sqlite"),
    )

    original = rtl._STAGES
    rtl._STAGES = (("detect", SimpleNamespace(run=stub_run)),)
    try:
        runtime.run(request)
    finally:
        rtl._STAGES = original

    finished = [
        e for e in telem.events
        if e.kind == "stage_finished" and e.payload.get("stage") == "detect"
    ]
    assert finished, "no stage_finished event recorded for detect"
    assert finished[0].payload["detections"] == 9


def test_runtime_uses_empty_metadata_when_stage_emits_no_metrics(tmp_path):
    import card_capture.pipeline.runtime_local as rtl

    def stub_run(state, **kwargs):
        pass  # emits nothing

    telem = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telem)
    request = PipelineRunRequest(
        run_id="r2",
        input_video="artifact://local/v.mov",
        output_root=f"artifact://local/{tmp_path}/out/",
        runtime_mode="cpu_debug",
        config={},
        db_path=str(tmp_path / "cards.sqlite"),
    )

    original = rtl._STAGES
    rtl._STAGES = (("detect", SimpleNamespace(run=stub_run)),)
    try:
        runtime.run(request)
    finally:
        rtl._STAGES = original

    finished = [
        e for e in telem.events
        if e.kind == "stage_finished" and e.payload.get("stage") == "detect"
    ]
    assert finished
    # No metrics emitted → metadata is empty → payload has only stage + elapsed_ms.
    assert set(finished[0].payload) == {"stage", "elapsed_ms"}
