"""CoreML backend for YOLOv8-OBB.

Leverages the Apple Neural Engine (ANE) for high-speed inference on
Apple Silicon. Fallback to PyTorch/MPS if CoreML package is missing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image

try:
    import coremltools as ct
except ImportError:
    ct = None


class CoreMLDetector:
    """Wrapper around YOLOv8-OBB CoreML export."""

    def __init__(
        self,
        model_path: str | Path,
        compute_units: str = "all", # all, cpu_and_gpu, cpu_only, cpu_and_ane
    ):
        if ct is None:
            raise ImportError("coremltools not installed")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"CoreML model not found at {model_path}")

        print(f"Loading CoreML model from {model_path} (units={compute_units})...")
        
        # Determine compute units
        units = ct.ComputeUnit.ALL
        if compute_units == "cpu_only":
            units = ct.ComputeUnit.CPU_ONLY
        elif compute_units == "cpu_and_gpu":
            units = ct.ComputeUnit.CPU_AND_GPU
        elif compute_units == "cpu_and_ane":
            units = ct.ComputeUnit.CPU_AND_ANE

        self.model = ct.models.MLModel(str(self.model_path), compute_units=units)

    def predict(self, image: Image.Image) -> Any:
        """Run inference on a single image."""
        # Note: Input names depend on the export config (usually 'image')
        # We assume the standard Ultralytics CoreML export shape.
        input_name = self.model.input_description[0].name
        
        # Resize to model expected size (usually 640x640)
        # TODO: dynamically read from model description
        
        res = self.model.predict({input_name: image})
        return res
