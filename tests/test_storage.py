import json
from pathlib import Path

from card_capture.models import CornerDetection
from card_capture.storage import Storage


def test_storage_v21_records_instance_view_and_evidence(tmp_path: Path):
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("/videos/input.mov", "hash", 1000, 1920, 1080)

    instance_id = storage.add_card_instance(video_id=video_id, track_id="card_1")
    view_id = storage.add_card_view(
        card_instance_id=instance_id,
        frame_index=12,
        timestamp_ms=400,
        detection=CornerDetection(
            corners=((0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)),
            confidence=0.8,
            metadata={"model": "fake"},
        ),
        rectified_path="output/rectified/card_1.jpg",
        quality_score={"sharpness": 0.92},
        is_canonical=True,
    )
    evidence_id = storage.add_evidence_frame(
        card_view_id=view_id,
        source_frame_path="output/frames/f12.jpg",
        frame_width=1920,
        frame_height=1080,
        metrics={"blur": 90.0},
    )

    assert instance_id == 1
    assert view_id == 1
    assert evidence_id == 1

    rows = storage.list_card_instances(video_id)
    assert len(rows) == 1
    assert rows[0]["id"] == instance_id
    assert rows[0]["video_id"] == video_id
    assert rows[0]["track_id"] == "card_1"


def test_storage_v21_serializes_card_view_and_evidence_json(tmp_path: Path):
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("/videos/input.mov", "hash", 1000, 1920, 1080)
    instance_id = storage.add_card_instance(video_id=video_id, track_id="card_2")
    view_id = storage.add_card_view(
        card_instance_id=instance_id,
        frame_index=24,
        timestamp_ms=800,
        detection=CornerDetection(
            corners=((1.0, 2.0), (11.0, 2.0), (11.0, 22.0), (1.0, 22.0)),
            confidence=0.95,
            metadata={"backend": "onnx", "version": "2.1"},
        ),
    )
    storage.add_evidence_frame(
        card_view_id=view_id,
        source_frame_path="output/frames/f24.jpg",
        frame_width=1280,
        frame_height=720,
        metrics={"brightness": 0.75, "contrast": 0.6},
    )

    with storage._connect() as conn:
        view_row = conn.execute(
            """
            SELECT corners_json, confidence, metadata_json, rectified_path,
                   quality_score_json, is_canonical
            FROM card_views
            WHERE id = ?
            """,
            (view_id,),
        ).fetchone()
        evidence_row = conn.execute(
            "SELECT metrics_json FROM evidence_frames WHERE card_view_id = ?",
            (view_id,),
        ).fetchone()

    assert view_row is not None
    assert json.loads(view_row["corners_json"]) == [
        [1.0, 2.0],
        [11.0, 2.0],
        [11.0, 22.0],
        [1.0, 22.0],
    ]
    assert view_row["confidence"] == 0.95
    assert json.loads(view_row["metadata_json"]) == {"backend": "onnx", "version": "2.1"}
    assert view_row["rectified_path"] is None
    assert view_row["quality_score_json"] is None
    assert view_row["is_canonical"] == 0
    assert evidence_row is not None
    assert json.loads(evidence_row["metrics_json"]) == {"brightness": 0.75, "contrast": 0.6}


def test_storage_records_performance_telemetry(tmp_path: Path):
    from card_capture.models import PerformanceTelemetry

    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("/videos/input.mov", "hash", 1000, 1920, 1080)

    telemetry = PerformanceTelemetry(
        t_ingest=1.0, t_detect=2.0, t_refine=3.0, t_io=4.0, queue_wait=5.0
    )

    storage.add_performance_log(video_id=video_id, frame_index=10, telemetry=telemetry)

    with storage._connect() as conn:
        row = conn.execute(
            "SELECT * FROM performance_logs WHERE video_id = ?", (video_id,)
        ).fetchone()

    assert row is not None
    assert row["frame_index"] == 10
    assert row["t_ingest"] == 1.0
    assert row["t_detect"] == 2.0
    assert row["t_refine"] == 3.0
    assert row["t_io"] == 4.0
    assert row["queue_wait"] == 5.0


def test_storage_supports_deduplication(tmp_path: Path):
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("/videos/input.mov", "hash", 1000, 1920, 1080)

    instance_id_1 = storage.add_card_instance(video_id=video_id, track_id="card_1")
    instance_id_2 = storage.add_card_instance(video_id=video_id, track_id="card_2")

    visual_hash_1 = "hash_1"
    visual_hash_2 = "hash_2"

    storage.update_instance_deduplication(instance_id_1, visual_hash_1)
    storage.update_instance_deduplication(
        instance_id_2, visual_hash_2, duplicate_of_id=instance_id_1
    )

    with storage._connect() as conn:
        row1 = conn.execute(
            "SELECT visual_hash, is_duplicate_of FROM card_instances WHERE id = ?",
            (instance_id_1,),
        ).fetchone()
        row2 = conn.execute(
            "SELECT visual_hash, is_duplicate_of FROM card_instances WHERE id = ?",
            (instance_id_2,),
        ).fetchone()

    assert row1["visual_hash"] == visual_hash_1
    assert row1["is_duplicate_of"] is None
    assert row2["visual_hash"] == visual_hash_2
    assert row2["is_duplicate_of"] == instance_id_1


def test_storage_find_canonical_for_hashes_uses_best_orientation(tmp_path: Path):
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("/videos/input.mov", "hash", 1000, 1920, 1080)
    instance_id = storage.add_card_instance(video_id=video_id, track_id="card_1")
    # canonical hash in DB
    canonical_hash = "00000000000000ff"
    storage.update_instance_deduplication(instance_id, canonical_hash)

    # first candidate is far; second candidate is a close rotational equivalent
    found = storage.find_canonical_for_hashes(
        ["ffffffffffffffff", "00000000000000f8"], threshold=6
    )
    assert found == instance_id
