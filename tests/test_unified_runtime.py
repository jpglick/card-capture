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
from card_capture.data.sql_queries import STORAGE_INIT_SCHEMA
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


# Golden-video smoke test. Skipped by default (the 234MB clip is not in the
# repo). The synthetic-fixture regression guard lives in
# ``tests/pipeline/test_back_half_e2e.py`` and runs in ~3s; this test exists
# for explicit re-validation against the production-shape input.
@pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="Golden-set video IMG_5872.MOV not present (large binary, not in repo)",
)
@pytest.mark.quarantine
def test_local_runtime_runs_to_completion(tmp_path: Path) -> None:
    db_path = tmp_path / "cards.sqlite"
    # Use the canonical storage schema so the store stage finds every table
    # (card_instances, card_views, saved_cards, track_telemetry,
    # pipeline_events, videos). Production migrations are exercised in
    # tests/data/test_*_repository.py.
    conn = open_connection(db_path)
    conn.executescript(STORAGE_INIT_SCHEMA)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs(
            run_id TEXT PRIMARY KEY,
            video_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            cards_extracted INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            finished_at TEXT
        );
        """
    )
    # Insert a real videos row so the store stage's FK-constrained inserts
    # (card_instances.video_id -> videos.id) succeed.
    conn.execute(
        "INSERT INTO videos (id, source_path, file_hash, duration_ms, width, height, status) "
        "VALUES (1, ?, 'deadbeef', 0, 0, 0, 'processing')",
        (str(FIXTURE),),
    )
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
        db_path=str(db_path),
        video_id=1,
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

    # Phase 10 — back-half is wired; expect at least one persisted card.
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM card_instances").fetchone()[0]
    assert count >= 1, "back-half stages did not produce any cards"
