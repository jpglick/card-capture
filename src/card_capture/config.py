import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class DebugConfig:
    export_frames: bool = False

@dataclass
class PipelineConfig:
    detector: str = "docaligner"
    reader_backend: str = "auto"
    queue_size: int = 256
    inference_batch_size: int = 16
    corner_confidence: float = 0.5
    blur_threshold: float = 30.0
    variance_threshold: float = 20.0
    empty_pixel_threshold: float = 0.98
    detection_width: int = 640
    device: str = "auto"
    group_gap_ms: int = 300
    spatial_variance_threshold: float = 150.0
    telemetry_scope: str = "canonical"
    triage_keep_percentile: float = 0.05
    debug: DebugConfig = field(default_factory=DebugConfig)

def load_config(path: Path) -> PipelineConfig:
    if not path.exists():
        return PipelineConfig()
    with open(path, "r") as f:
        data = json.load(f)
    debug_data = data.pop("debug", {})
    debug_config = DebugConfig(**debug_data)
    return PipelineConfig(debug=debug_config, **data)

def save_config(config: PipelineConfig, path: Path) -> None:
    data = config.__dict__.copy()
    data["debug"] = config.debug.__dict__
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
