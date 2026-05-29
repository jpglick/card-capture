"""Smoke test for the V5.5 in-process pipeline runtime.

Historically this file targeted `UnifiedRuntime`. After the V5.5 refactor
the same role is filled by `LocalPipelineRuntime` (see
`src/card_capture/pipeline/runtime_local.py`). This test exercises a
synthetic-fixture run end-to-end against the new contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from card_capture.data.connection import open_connection
from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import InMemoryTelemetry


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "golden_corpus"
    / "IMG_5872"
    / "IMG_5872.MOV"
)


@pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="Golden-set video IMG_5872.MOV not present (large binary, not in repo)",
)
def test_local_runtime_runs_to_completion(tmp_path: Path) -> None:
    db_path = tmp_path / "cards.sqlite"
    # Synthetic schema sufficient for the smoke contract; production migrations
    # are exercised in tests/data/test_*_repository.py.
    conn = open_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pipeline_runs(
            run_id TEXT PRIMARY KEY,
            video_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            cards_extracted INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            finished_at TEXT
        );
        CREATE TABLE IF NOT EXISTS card_instances(
            id INTEGER PRIMARY KEY,
            instance_id TEXT UNIQUE,
            video_id INTEGER NOT NULL,
            run_id TEXT,
            track_id TEXT NOT NULL,
            fused_image_path TEXT
        );
    """)
    conn.commit()
    conn.close()

    telemetry = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    request = PipelineRunRequest(
        run_id="smoke-unified",
        input_video=f"artifact://local/{FIXTURE}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
        config={"db_path": str(db_path)},
    )

    result = runtime.run(request)

    assert result.manifest.runtime_mode == "cpu_debug"
    assert result.manifest.input_video == request.input_video

    # Stage facades must have fired.
    finished_stages = {
        e.payload["stage"] for e in telemetry.events if e.kind == "stage_finished"
    }
    expected = {
        "sample", "detect", "novelty", "track", "refine",
        "score", "resolve", "fuse", "dedup", "store",
    }
    missing = expected - finished_stages
    assert not missing, f"missing stage_finished events: {sorted(missing)}"
