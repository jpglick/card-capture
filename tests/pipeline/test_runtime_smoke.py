"""LocalPipelineRuntime runs all stages in one process and produces a manifest.

Phase 3 smoke test: uses a tiny synthetic video fixture. Real-video tests
go in tests/performance/.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import InMemoryTelemetry


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_synthetic.MOV"


@pytest.mark.skipif(not FIXTURE.exists(), reason="tiny_synthetic.MOV fixture not present")
def test_local_runtime_single_process(tmp_path):
    telemetry = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    req = PipelineRunRequest(
        run_id="smoke",
        input_video=f"artifact://local/{FIXTURE}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
    )
    result = runtime.run(req)

    # Models load at most once per run.
    model_loads = [e for e in telemetry.events if e.payload.get("event") == "model_load"]
    assert len(model_loads) <= 4  # YOLO + DINOv2 + tracker + (any future) — one each, not per stage

    # Video opens at most once.
    decode_opens = [e for e in telemetry.events if e.payload.get("event") == "decode_open"]
    assert len(decode_opens) == 1, f"expected exactly one decode_open, saw {len(decode_opens)}"

    # All known stages emitted.
    finished = {e.payload["stage"] for e in telemetry.events if e.kind == "stage_finished"}
    for stage in ("sample", "detect", "novelty", "track", "refine", "score", "resolve", "fuse", "dedup", "store"):
        assert stage in finished, f"stage {stage} missing"

    assert result.manifest.runtime_mode == "cpu_debug"
