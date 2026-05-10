from tests.regression.pipeline_runner import HarnessInstance, instances_from_db_rows


def test_instances_from_db_rows_maps_fields():
    rows = [
        {
            "instance_id": 7,
            "video_id": 3,
            "session_id": 2,
            "angle": "Front",
            "is_duplicate_of": None,
            "fused_image_path": "out/foo.jpg",
            "start_time": 12000,
            "end_time": 16000,
            "detection_count": 18,
            "phash": "abc",
        },
    ]
    out = instances_from_db_rows(rows)
    assert len(out) == 1
    inst = out[0]
    assert isinstance(inst, HarnessInstance)
    assert inst.instance_id == 7
    assert inst.angle == "Front"
    assert inst.start_ms == 12000
    assert inst.end_ms == 16000
    assert inst.duplicate_of is None
