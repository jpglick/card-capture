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

def test_storage_deduplication_updates(tmp_path: Path):
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("/videos/input.mov", "hash", 1000, 1920, 1080)

    # Create two instances
    instance1_id = storage.add_card_instance(video_id=video_id, track_id="card_1")
    instance2_id = storage.add_card_instance(video_id=video_id, track_id="card_2")

    # Update instance 1 with a hash
    visual_hash = "abcdef1234567890"
    storage.update_instance_deduplication(instance1_id, visual_hash=visual_hash)

    # Update instance 2 as a duplicate of instance 1
    storage.update_instance_deduplication(instance2_id, visual_hash=visual_hash, duplicate_of_id=instance1_id)

    # Verify updates in listing
    instances = storage.list_card_instances(video_id)
    assert len(instances) == 2

    i1 = next(i for i in instances if i["id"] == instance1_id)
    assert i1["visual_hash"] == visual_hash
    assert i1["is_duplicate_of"] is None

    i2 = next(i for i in instances if i["id"] == instance2_id)
    assert i2["visual_hash"] == visual_hash
    assert i2["is_duplicate_of"] == instance1_id
