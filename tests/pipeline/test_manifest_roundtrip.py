"""RunManifest must round-trip through JSON without provider-specific assumptions."""
from __future__ import annotations

import json
from pathlib import Path

from card_capture.pipeline.request import (
    PipelineRunRequest,
    PipelineRunResult,
    RunManifest,
    StageTiming,
    ContractViolation,
)


def test_manifest_roundtrip_minimal():
    manifest = RunManifest(
        run_id="abc123",
        runtime_mode="cpu_debug",
        input_video="artifact://local/in.MOV",
        output_artifacts=["artifact://local/abc123/cards/"],
        cards=[],
        stage_timings=[StageTiming(stage="sample", elapsed_ms=12.5)],
        contract_violations=[],
        version="0.5.5+phase1",
    )
    blob = manifest.to_json()
    again = RunManifest.from_json(blob)
    assert again == manifest


def test_request_serializable_only_references():
    req = PipelineRunRequest(
        run_id="abc123",
        input_video="artifact://local/in.MOV",
        output_root="artifact://local/abc123/",
        runtime_mode="cpu_debug",
        config={"corner_confidence": 0.5},
    )
    blob = json.dumps(req.to_dict())
    again = PipelineRunRequest.from_dict(json.loads(blob))
    assert again == req


def test_contract_violation_has_stable_code():
    v = ContractViolation(code="cpu_read_in_strict", metadata={"call_site": "foo:42"})
    assert v.code == "cpu_read_in_strict"
    assert v.metadata["call_site"] == "foo:42"
