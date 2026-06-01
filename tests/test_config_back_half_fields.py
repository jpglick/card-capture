"""Phase 1 — PipelineConfig has all back-half fields with V4 defaults."""
from card_capture.config import PipelineConfig


def test_pipeline_config_has_novelty_floor():
    assert PipelineConfig().novelty_floor == 0.30


def test_pipeline_config_has_track_confidence_floor():
    assert PipelineConfig().track_confidence_floor == 0.0


def test_pipeline_config_has_stand_novelty_max():
    assert PipelineConfig().stand_novelty_max == 0.065


def test_pipeline_config_has_stand_sharpness_max():
    assert PipelineConfig().stand_sharpness_max == 0.092


def test_pipeline_config_has_foil_threshold():
    assert PipelineConfig().foil_threshold == 50.0


def test_pipeline_config_has_enable_foil_aware_fusion():
    assert PipelineConfig().enable_foil_aware_fusion is True


def test_pipeline_config_has_use_fb_classifier():
    assert PipelineConfig().use_fb_classifier is True


def test_pipeline_config_has_laplacian_scan_stride():
    assert PipelineConfig().laplacian_scan_stride == 4


def test_pipeline_config_has_max_corner_gap_frames():
    assert PipelineConfig().max_corner_gap_frames == 15


def test_pipeline_config_has_corner_refinement():
    assert PipelineConfig().corner_refinement is False
