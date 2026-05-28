import pytest
import numpy as np
from pathlib import Path
from card_capture.runtime import UnifiedRuntime, PipelineRunRequest
from card_capture.detectors import FakeCardDetector
from card_capture.sampler import VideoSampler

@pytest.mark.quarantine
def test_unified_runtime_smoke(tmp_path):
    # Setup
    video_path = Path("tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV")
    if not video_path.exists():
        pytest.skip("Golden set video not found")
        
    output_dir = tmp_path / "output"
    db_path = tmp_path / "cards.sqlite"
    output_dir.mkdir()
    
    # We need to run migrations to have a valid DB schema
    from migrations.run_migrations import apply_migrations
    apply_migrations(db_path)
    
    # Initialize the database with the video record to satisfy FK constraints
    from card_capture.storage import Storage
    storage = Storage(str(db_path))
    video_id = storage.add_video(
        source_path=str(video_path),
        file_hash="fake-hash",
        duration_ms=1000,
        width=1280,
        height=720
    )
    
    from card_capture.sampler import StrideSampler
    sampler = StrideSampler(video_path=video_path, target_yolo_fps=1.0)
    # Use high confidence so they aren't filtered
    detector = FakeCardDetector(confidence_threshold=0.95)
    
    runtime = UnifiedRuntime(sampler, detector)
    
    request = PipelineRunRequest(
        video_path=video_path,
        output_dir=output_dir,
        db_path=db_path,
        video_id=video_id,
        detector_backend="fake",
        runtime_mode="cpu_debug"
    )
    
    # Act
    result = runtime.run(request)
    
    # Assert
    assert result.success is True
    assert result.processing_result is not None
    assert result.processing_result.saved_instance_count > 0
    assert db_path.exists()
    assert (output_dir / "crops").exists()
