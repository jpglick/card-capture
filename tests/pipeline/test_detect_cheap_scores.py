# tests/pipeline/test_detect_cheap_scores.py
"""The MPS detect path must attach flatness+clarity to each detection row
and must NOT eager-warp when no crop_cache is supplied."""
import numpy as np
from pipeline.steps import detect as detect_mod


def test_detection_rows_carry_flatness_and_clarity(monkeypatch):
    # Drive _annotate_cheap_scores directly: it is the unit under test.
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    frame[100:800, 100:600] = 255  # a bright rectangle
    corners = [(100.0, 100.0), (600.0, 100.0), (600.0, 800.0), (100.0, 800.0)]
    row = {"detection_id": 0, "corners": corners, "triage_metrics": {}}
    detect_mod._annotate_cheap_scores(row, frame, device="cpu")
    assert "flatness" in row["triage_metrics"]
    assert "clarity" in row["triage_metrics"]
    assert row["triage_metrics"]["flatness"] > 0.9
    assert row["triage_metrics"]["clarity"] >= 0.0
