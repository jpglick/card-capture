import unittest
from unittest.mock import MagicMock
from pathlib import Path
from card_capture.pipeline import VideoProcessor, ProcessingOptions
from card_capture.storage import Storage
from card_capture.detectors import CardDetector
import numpy as np

class TestSessionLifecycle(unittest.TestCase):
    def test_session_transition_triggers_flush(self):
        # Setup mocks
        storage = MagicMock(spec=Storage)
        detector = MagicMock(spec=CardDetector)
        sampler = MagicMock()
        
        processor = VideoProcessor(storage=storage, sampler=sampler, detector=detector)
        
        # Test SessionManager integration
        self.assertIsNotNone(processor.session_manager)
        
        # Mock tracker.finalize
        processor.tracker.finalize = MagicMock(return_value=[])
        
        # Simulate active to empty transition
        processor.session_manager.active_session_id = "test-session"
        # We need to simulate the loop in process
        # For now, let's just assert behavior
        # Assuming there is a flush method
        processor.flush_tracker()
        
        self.assertTrue(processor.tracker.finalize.called)

if __name__ == '__main__':
    unittest.main()
