
import numpy as np
from pathlib import Path
from card_capture.pipeline import NullStateDetector, _producer_main, _ProducerStats
from card_capture.models import FrameSample
import multiprocessing as mp
from queue import Empty

class MockSampler:
    def __init__(self):
        self.background_proxies = []
        self.called_sample = False

    def sample(self, video_path: Path, sample_fps: float):
        self.called_sample = True
        # Simulate background_proxies being populated AFTER sample is called (or during first iteration)
        self.background_proxies = [np.zeros((100, 100, 3), dtype=np.uint8)]
        yield FrameSample(frame_index=0, timestamp_ms=0, image=np.zeros((100, 100, 3), dtype=np.uint8), width=100, height=100)

def test_producer_missing_proxies():
    # This test demonstrates the bug where proxies are passed empty
    sampler = MockSampler()
    video_path = "mock.mp4"
    frame_queue = mp.Queue()
    stats_queue = mp.Queue()
    error_queue = mp.Queue()
    
    # In the current (buggy) implementation, proxies are extracted BEFORE starting producer
    background_proxies = getattr(sampler, "background_proxies", [])
    assert len(background_proxies) == 0
    
    # We can't easily run _producer_main in a separate process here and check its internal state 
    # without mocking more stuff, but we can see that if we pass background_proxies here, 
    # they will be empty.
    
    # Let's mock what _producer_main does
    null_detector = NullStateDetector(frames=30, threshold=15.0)
    if background_proxies:
        null_detector.warmup_batch(background_proxies)
    
    assert null_detector.background_model is None
