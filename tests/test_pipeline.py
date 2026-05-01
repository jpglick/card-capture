from pathlib import Path

import numpy as np

from card_capture.models import CardDetection, FrameSample
from card_capture.pipeline import ProcessingOptions, VideoProcessor
from card_capture.storage import Storage


class FakeSampler:
    def sample(self, video_path, sample_fps):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:90, 10:90] = 180
        yield FrameSample(
            frame_index=1,
            timestamp_ms=100,
            image=image,
            width=100,
            height=100,
        )


class FakeDetector:
    runtime = "fake"
    model_name = "fake-card-detector"

    def detect(self, frame):
        return [
            CardDetection(
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                polygon=((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)),
                confidence=0.95,
            )
        ]


def test_video_processor_saves_best_crop_and_metadata(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake video content")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()

    result = VideoProcessor(
        storage=storage,
        sampler=FakeSampler(),
        detector=FakeDetector(),
    ).process(
        video_path,
        ProcessingOptions(output_dir=tmp_path / "output", sample_fps=5, max_candidates=5),
    )

    saved_cards = storage.list_saved_cards()

    assert result.saved_count == 1
    assert len(saved_cards) == 1
    assert Path(saved_cards[0]["image_path"]).exists()
    assert saved_cards[0]["final_score"] > 0
