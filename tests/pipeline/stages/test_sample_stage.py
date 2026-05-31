"""sample stage starts a FrameProducer and does not drain it."""
from unittest.mock import MagicMock

from card_capture.pipeline.stages import sample as sample_stage
from card_capture.sampler.frame_producer import FrameProducer


def test_sample_starts_producer_without_draining(synthetic_two_cards_mov):
    request = MagicMock()
    request.input_video = str(synthetic_two_cards_mov)
    state = {"request": request}

    sample_stage.run(state, telemetry=MagicMock())

    assert isinstance(state["frame_producer"], FrameProducer)
    assert state["sampled_frames"] == []
    assert state["estimated_frame_total"] > 0
    assert state["video_path"] == str(synthetic_two_cards_mov)

    frames = list(state["frame_producer"])
    assert len(frames) > 0
    assert [f.frame_index for f in frames] == sorted(f.frame_index for f in frames)
