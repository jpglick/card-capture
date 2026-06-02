"""Phase 10 — synthetic e2e: cards > 0 after a full run."""
from pathlib import Path

import pytest

from card_capture.pipeline.request import PipelineRunRequest
from card_capture.pipeline.runtime_local import LocalPipelineRuntime
from card_capture.pipeline.telemetry import InMemoryTelemetry


def _init_db(path: Path) -> None:
    """Build the test DB from the REAL production schema.

    Using ``STORAGE_INIT_SCHEMA`` (the same DDL production runs) keeps this
    fixture from drifting out of sync with the actual ``card_instances`` /
    ``card_views`` / ``saved_cards`` / ``track_telemetry`` / ``pipeline_events``
    columns and their FK/NOT NULL constraints — the exact drift that previously
    let column-name bugs through. ``pipeline_runs`` is created by migrations in
    production (not STORAGE_INIT_SCHEMA), so it is added explicitly here.
    """
    import sqlite3

    from card_capture.stages.store.storage import Storage

    # Creates videos, card_instances, card_views, saved_cards, track_telemetry,
    # pipeline_events with the real columns + FK constraints.
    Storage(path).initialize()

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY, video_id INTEGER, status TEXT,
                cards_extracted INTEGER DEFAULT 0, started_at TEXT, finished_at TEXT
            )
            """
        )
        # FK parent for card_instances / saved_cards / track_telemetry / pipeline_events.
        conn.execute(
            "INSERT INTO videos (id, source_path, file_hash, duration_ms, width, height) "
            "VALUES (1, '/tmp/synthetic.mov', 'deadbeef', 4000, 750, 1050)"
        )


def test_back_half_e2e_produces_cards(synthetic_two_cards_mov, tmp_path):
    db = tmp_path / "cards.sqlite"
    _init_db(db)

    # Register the run
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, video_id, status) VALUES (?, ?, ?)",
            ("e2e-1", 1, "processing")
        )

    telemetry = InMemoryTelemetry()
    runtime = LocalPipelineRuntime(telemetry=telemetry)
    req = PipelineRunRequest(
        run_id="e2e-1",
        input_video=f"artifact://local/{synthetic_two_cards_mov}",
        output_root=f"artifact://local/{tmp_path}/",
        runtime_mode="cpu_debug",
        config={
            "detector": "fake",   # Synthesises 2 corner detections per frame
            "device": "cpu",
            "use_kornia": True,
            "kornia_device": "cpu",
            "rotate_180": False,
            "tracker_backend": "bytetrack",
            "min_track_length": 1,
            "fusion_target_frames": 1,
            "novelty_floor": 0.0,
            "track_confidence_floor": 0.0,
            "stand_novelty_max": 0.0,
            "stand_sharpness_max": 0.0,
            "use_fb_classifier": False,
            "enable_foil_aware_fusion": False,
            "laplacian_scan_stride": 0,
            "max_corner_gap_frames": 30,
            "corner_refinement": False,
        },
        db_path=str(db),
        video_id=1,
    )
    result = runtime.run(req)

    # Stages all fired
    finished = {e.payload["stage"] for e in telemetry.events if e.kind == "stage_finished"}
    expected = {"sample", "detect", "novelty", "track", "refine",
                "score", "resolve", "fuse", "dedup", "store"}
    assert expected <= finished, f"missing stages: {expected - finished}"

    # At least one card persisted
    import sqlite3
    with sqlite3.connect(db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM card_instances WHERE run_id=?", ("e2e-1",)
        ).fetchone()[0]
    assert count >= 1, "store stage did not persist any card_instances"

    # crops/ has fused images
    crops = list((tmp_path / "crops").glob("instance_*_fused.jpg"))
    assert len(crops) >= 1

    # Run marked completed
    with sqlite3.connect(db) as conn:
        status, cards = conn.execute(
            "SELECT status, cards_extracted FROM pipeline_runs WHERE run_id=?", ("e2e-1",)
        ).fetchone()
    assert status == "completed"
    assert cards >= 1
