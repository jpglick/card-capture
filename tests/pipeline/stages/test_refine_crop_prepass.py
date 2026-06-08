import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from card_capture.core.config import PipelineConfig
from card_capture.stages import refine
from card_capture.stages.refine.cropper import PrecisionNormalizer


def test_config_exposes_crop_knobs_in_request_dict():
    cfg = PipelineConfig()
    d = cfg.to_request_config()
    assert d["refine_crop_margin_px"] == 8
    assert d["refine_min_available_mb"] == 2048.0
