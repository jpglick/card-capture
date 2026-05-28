"""Assert that refine does not re-open the source video.

Uses InMemoryTelemetry to count `decode_open` events.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import InMemoryTelemetry


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_synthetic.MOV"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_refine_does_not_redecode(tmp_path):
    telemetry = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    req = PipelineRunRequest(
        run_id="r",
        input_video=f"artifact://local/{FIXTURE}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
    )
    runtime.run(req)

    opens = [
        e for e in telemetry.events
        if e.kind == "resource_sample" and e.payload.get("event") == "decode_open"
    ]
    assert len(opens) == 1, f"expected exactly 1 decode_open, saw {len(opens)}: {opens}"
