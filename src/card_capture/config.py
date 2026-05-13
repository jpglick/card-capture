import json
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class DebugConfig:
    export_frames: bool = False

@dataclass
class PipelineConfig:
    # Core Orchestration
    detector: str = "docaligner"
    reader_backend: str = "auto"
    queue_size: int = 256
    inference_batch_size: int = 16
    device: str = "auto"
    
    # Detection & Quality
    corner_confidence: float = 0.5
    blur_threshold: float = 30.0
    variance_threshold: float = 20.0
    empty_pixel_threshold: float = 0.98
    detection_width: int = 640
    
    # Presence & Sampling
    background_frames: int = 30
    background_threshold: float = 15.0
    null_patience_frames: int = 20
    triage_keep_percentile: float = 0.05
    fast_scan_fps: float = 15.0
    confirm_scan_fps: float = 5.0
    valley_drop_ratio: float = 0.40
    valley_min_width_frames: int = 3
    delta_spike_ratio: float = 0.50
    
    # Tracking
    tracker_backend: str = "bytetrack"
    group_gap_ms: int = 300
    spatial_variance_threshold: float = 300.0  # Consolidating to monolith's 300
    min_track_length: int = 6                 # Consolidating to monolith's 6
    centroid_jump_ratio: float = 0.30
    centroid_jump_frames: int = 3
    
    # Post-Processing
    rotate_180: bool = False
    reid_distance_threshold: float = 0.6
    fusion_target_frames: int = 4
    foil_threshold: float = 50.0
    enable_foil_aware_fusion: bool = True
    corner_refinement: bool = False
    
    # Hardware Acceleration
    use_kornia: bool = True
    telemetry_scope: str = "canonical"
    
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_options(self, output_dir: Path) -> "card_capture.pipeline.ProcessingOptions":
        """Convert to legacy ProcessingOptions for the monolith path."""
        from .pipeline import ProcessingOptions
        return ProcessingOptions(
            output_dir=output_dir,
            reader_backend=self.reader_backend,
            queue_size=self.queue_size,
            inference_batch_size=self.inference_batch_size,
            corner_confidence_threshold=self.corner_confidence,
            blur_threshold=self.blur_threshold,
            variance_threshold=self.variance_threshold,
            empty_pixel_threshold=self.empty_pixel_threshold,
            group_gap_ms=self.group_gap_ms,
            spatial_variance_threshold=self.spatial_variance_threshold,
            telemetry_scope=self.telemetry_scope,
            background_frames=self.background_frames,
            background_threshold=self.background_threshold,
            null_patience_frames=self.null_patience_frames,
            min_track_length=self.min_track_length,
            use_kornia=self.use_kornia,
            kornia_device=self.device,
            triage_keep_percentile=self.triage_keep_percentile,
            rotate_180=self.rotate_180,
            tracker_backend=self.tracker_backend,
            centroid_jump_ratio=self.centroid_jump_ratio,
            centroid_jump_frames=self.centroid_jump_frames,
            foil_threshold=self.foil_threshold,
            enable_foil_aware_fusion=self.enable_foil_aware_fusion,
        )

def load_config(path: Path) -> PipelineConfig:
    if not path.exists():
        return PipelineConfig()
    with open(path, "r") as f:
        data = json.load(f)
    debug_data = data.pop("debug", {})
    debug_config = DebugConfig(**debug_data)
    
    # Filter data to only include known fields
    known_fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    data = {k: v for k, v in data.items() if k in known_fields}
    
    return PipelineConfig(debug=debug_config, **data)

def save_config(config: PipelineConfig, path: Path) -> None:
    data = config.__dict__.copy()
    data["debug"] = config.debug.__dict__
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
