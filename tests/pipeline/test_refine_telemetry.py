import numpy as np

from pipeline.steps.refine import _describe_kornia_batch


def test_describe_kornia_batch_records_shapes_and_memory():
    batch_items = [
        (np.zeros((2160, 3840, 3), dtype=np.uint8), []),
        (np.zeros((1080, 1920, 3), dtype=np.uint8), []),
        ("missing.jpg", []),
    ]

    result = _describe_kornia_batch(
        batch_items,
        batch_ids=[1, 2, 3],
        elapsed_s=1.2345,
        device="cuda",
        memory_before={"allocated_mb": 100.0, "reserved_mb": 200.0},
        memory_after={"allocated_mb": 130.5, "reserved_mb": 260.5},
    )

    assert result["batch_size"] == 3
    assert result["detection_ids"] == [1, 2, 3]
    assert result["elapsed_ms"] == 1234.5
    assert result["device"] == "cuda"
    assert result["input_shapes"] == [
        {"shape": [1080, 1920, 3], "count": 1},
        {"shape": [2160, 3840, 3], "count": 1},
        {"shape": None, "count": 1},
    ]
    assert result["input_pixels_total"] == 10_368_000
    assert result["cuda_memory_before_mb"] == {"allocated_mb": 100.0, "reserved_mb": 200.0}
    assert result["cuda_memory_after_mb"] == {"allocated_mb": 130.5, "reserved_mb": 260.5}
    assert result["cuda_memory_delta_mb"] == {"allocated_mb": 30.5, "reserved_mb": 60.5}
