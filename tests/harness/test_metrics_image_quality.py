"""Tests for harness.metrics.image_quality."""
import pytest
from pathlib import Path

from harness.metrics.image_quality import ImageQuality, image_quality

FIX = Path("tests/harness/fixtures/runs")


def test_image_quality_identical_images():
    """Fixture: fused images and reference frames are byte-identical → SSIM ≈ 1.0."""
    result = image_quality(
        db_path=FIX / "image_quality" / "cards.sqlite",
        truth_path=FIX / "image_quality" / "truth.json",
        video_id="image_quality",
    )
    assert isinstance(result, ImageQuality)
    assert result.mean_ssim is not None
    assert result.mean_ssim == pytest.approx(1.0, abs=1e-4)
    assert result.coverage == pytest.approx(1.0, abs=1e-6)


def test_image_quality_psnr_reported():
    """PSNR is reported (not None) when images are identical."""
    result = image_quality(
        db_path=FIX / "image_quality" / "cards.sqlite",
        truth_path=FIX / "image_quality" / "truth.json",
        video_id="image_quality",
    )
    # PSNR of identical images is effectively infinite; skimage returns ~inf or very large
    assert result.mean_psnr is not None


def test_image_quality_none_when_no_reference_frames(tmp_path):
    """No reference frames directory → mean_ssim is None, coverage = 0."""
    truth_path = tmp_path / "t.json"
    truth_path.write_text(
        '{"video_id":"image_quality","schema_version":1,"expected_cards":['
        '{"card_id":"card_01","front_present":true,"back_present":false,'
        '"physical_card_key":"key_01","is_foil":false,'
        '"approx_front_window_ms":[1000,2000]}]}'
    )
    # truth_path.parent has no "reference_frames" subdirectory
    result = image_quality(
        db_path=FIX / "image_quality" / "cards.sqlite",
        truth_path=truth_path,
        video_id="image_quality",
    )
    assert result.mean_ssim is None
    assert result.mean_psnr is None
    assert result.coverage == pytest.approx(0.0, abs=1e-6)


def test_image_quality_result_is_frozen():
    """ImageQuality instances are immutable."""
    iq = ImageQuality(mean_ssim=0.9, mean_psnr=30.0, coverage=1.0)
    with pytest.raises(Exception):
        iq.mean_ssim = 0.0  # type: ignore[misc]
