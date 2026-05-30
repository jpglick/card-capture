"""sample+detect overlap: frame parity vs a direct sampler pass; one det/frame."""
from unittest.mock import MagicMock

from card_capture.pipeline.stages import detect as detect_stage
from card_capture.pipeline.stages import sample as sample_stage
from card_capture.sampler import StrideSampler


def test_sample_detect_overlap_parity(synthetic_two_cards_mov):
    request = MagicMock()
    request.input_video = str(synthetic_two_cards_mov)
    request.config = {"detector": "fake"}
    state = {"request": request}

    sample_stage.run(state, telemetry=MagicMock())
    detect_stage.run(state, telemetry=MagicMock())

    sampled = state["sampled_frames"]
    detections = state["detections"]

    expected = list(StrideSampler(video_path=synthetic_two_cards_mov).sample())
    assert len(sampled) > 0
    assert [f.frame_index for f in sampled] == [f.frame_index for f in expected]
    assert [f.timestamp_ms for f in sampled] == [f.timestamp_ms for f in expected]

    assert len(detections) == len(sampled)
    assert [d["frame_index"] for d in detections] == [f.frame_index for f in sampled]
