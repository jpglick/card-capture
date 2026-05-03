import numpy as np
import pytest
from pathlib import Path
from card_capture.pipeline import VideoProcessor, ProcessingOptions
from card_capture.storage import Storage
from card_capture.models import FramePacket, DetectionPacket, CornerDetection, FrameSample

class FakeSampler:
    def __init__(self, frames):
        self.frames = frames
    def sample(self, video_path, sample_fps):
        yield from self.frames

class FakeDetector:
    def __init__(self, confidence=0.95):
        self.confidence = confidence
    def detect_batch(self, frames, confidence_threshold):
        detections = []
        for frame in frames:
            # Shift corners slightly for each frame to simulate tracking
            detections.append(DetectionPacket(
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                width=frame.width,
                height=frame.height,
                corner_detection=CornerDetection(
                    corners=((10.0 + frame.frame_index, 10.0), (90.0 + frame.frame_index, 10.0), (90.0 + frame.frame_index, 90.0), (10.0 + frame.frame_index, 90.0)),
                    confidence=self.confidence
                )
            ))
        return detections

def test_pipeline_refinery_phase(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    storage = Storage(db_path)
    storage.initialize()
    
    output_dir = tmp_path / "output"
    
    # Create synthetic frames (identical content for deduplication)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[20:180, 20:180] = 150 # Gray card
    
    frames = [
        FrameSample(frame_index=0, timestamp_ms=0, image=image.copy(), width=200, height=200),
        FrameSample(frame_index=1, timestamp_ms=100, image=image.copy(), width=200, height=200)
    ]
    
    processor = VideoProcessor(storage=storage, sampler=FakeSampler(frames), detector=FakeDetector())
    
    # Process first video
    video_path1 = tmp_path / "video1.mov"
    video_path1.write_bytes(b"v1")
    processor.process(video_path1, ProcessingOptions(output_dir=output_dir / "v1"))
    
    instances = storage.list_card_instances(video_id=1)
    assert len(instances) == 1
    v1_hash = instances[0]["visual_hash"]
    assert v1_hash is not None
    assert instances[0]["angle"] == "Front"
    
    # Process second video with same card (should be duplicate)
    video_path2 = tmp_path / "video2.mov"
    video_path2.write_bytes(b"v2")
    processor.process(video_path2, ProcessingOptions(output_dir=output_dir / "v2"))
    
    instances2 = storage.list_card_instances(video_id=2)
    assert len(instances2) == 1
    assert instances2[0]["visual_hash"] == v1_hash
    assert instances2[0]["is_duplicate_of"] == instances[0]["id"]
    
    # Verify crops exist for ALL canonical views
    v2_crops = list((output_dir / "v2" / "crops").glob("*.jpg"))
    # frames_per_instance=2 default, so 2 crops per instance
    assert len(v2_crops) == 2
