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
    config_preset: str = "balanced"
    queue_size: int = 256
    inference_batch_size: int = 16
    device: str = "auto"
    
    # Detection & Quality
    corner_confidence: float = 0.4
    blur_threshold: float = 30.0
    variance_threshold: float = 20.0
    empty_pixel_threshold: float = 0.98
    detection_width: int = 640
    
    # Presence & Sampling
    presence_threshold: float = 0.4
    background_frames: int = 30
    background_threshold: float = 15.0
    null_patience_frames: int = 30
    triage_keep_percentile: float = 0.05
    fast_scan_fps: float = 15.0
    confirm_scan_fps: float = 5.0
    target_yolo_fps: float = 3.0
    opening_scan_s: float = 2.0
    valley_drop_ratio: float = 0.40
    valley_min_width_frames: int = 3
    delta_spike_ratio: float = 0.50

    # ------------------------------------------------------------------
    # V5.5 back-half stage knobs (see docs/.../2026-05-29-v55-back-half-spec.md §4.8)
    # ------------------------------------------------------------------
    novelty_floor: float = 0.30
    track_confidence_floor: float = 0.0
    stand_novelty_max: float = 0.065
    stand_sharpness_max: float = 0.092
    foil_threshold: float = 50.0
    enable_foil_aware_fusion: bool = True
    use_fb_classifier: bool = True
    laplacian_scan_stride: int = 4
    max_corner_gap_frames: int = 15
    corner_refinement: bool = False
    
    # Tracking
    tracker_backend: str = "botsort"
    group_gap_ms: int = 300
    spatial_variance_threshold: float = 300.0  # Consolidating to monolith's 300
    min_track_length: int = 3
    centroid_jump_ratio: float = 0.30
    centroid_jump_frames: int = 3
    # Tuned on IMG_5922 (26 fronts / 18 holder bridges) via scripts/diag_swap_signals.py.
    appearance_same_threshold: float = 0.16
    appearance_change_threshold: float = 0.16
    appearance_confirm_frames: int = 3
    bridge_min_occurrences: int = 3
    bridge_position_ratio: float = 0.80
    bridge_neighbor_change_ratio: float = 0.80
    bridge_novelty_margin: float = 0.05
    bridge_max_length_ratio: float = 0.75
    
    # Post-Processing
    rotate_180: bool = False
    fusion_target_frames: int = 1
    
    # Hardware Acceleration
    use_kornia: bool = True
    telemetry_scope: str = "canonical"
    cuda_stride: int = 2
    cuda_batch_size: int = 32
    
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_request_config(self) -> dict:
        """Return the subset of fields back-half stages consume via
        `PipelineRunRequest.config`. Stays JSON-serializable so the
        request can cross the runtime/transport boundary unchanged."""
        return {
            # Detector / device / detection
            "detector": self.detector,
            "device": self.device,
            "corner_confidence": self.corner_confidence,
            "detection_width": self.detection_width,
            # Sampler
            "presence_threshold": self.presence_threshold,
            "fast_scan_fps": self.fast_scan_fps,
            "confirm_scan_fps": self.confirm_scan_fps,
            "valley_drop_ratio": self.valley_drop_ratio,
            "valley_min_width_frames": self.valley_min_width_frames,
            "delta_spike_ratio": self.delta_spike_ratio,
            # Tracker
            "tracker_backend": self.tracker_backend,
            "min_track_length": self.min_track_length,
            "centroid_jump_ratio": self.centroid_jump_ratio,
            "centroid_jump_frames": self.centroid_jump_frames,
            "appearance_same_threshold": self.appearance_same_threshold,
            "appearance_change_threshold": self.appearance_change_threshold,
            "appearance_confirm_frames": self.appearance_confirm_frames,
            "bridge_min_occurrences": self.bridge_min_occurrences,
            "bridge_position_ratio": self.bridge_position_ratio,
            "bridge_neighbor_change_ratio": self.bridge_neighbor_change_ratio,
            "bridge_novelty_margin": self.bridge_novelty_margin,
            "bridge_max_length_ratio": self.bridge_max_length_ratio,
            # Refine / fusion
            "fusion_target_frames": self.fusion_target_frames,
            "rotate_180": self.rotate_180,
            "use_kornia": self.use_kornia,
            "kornia_device": self.device,
            # New back-half knobs
            "novelty_floor": self.novelty_floor,
            "track_confidence_floor": self.track_confidence_floor,
            "stand_novelty_max": self.stand_novelty_max,
            "stand_sharpness_max": self.stand_sharpness_max,
            "foil_threshold": self.foil_threshold,
            "enable_foil_aware_fusion": self.enable_foil_aware_fusion,
            "use_fb_classifier": self.use_fb_classifier,
            "laplacian_scan_stride": self.laplacian_scan_stride,
            "max_corner_gap_frames": self.max_corner_gap_frames,
            "corner_refinement": self.corner_refinement,
        }

    def to_options(self, output_dir: Path) -> "card_capture.workers.ProcessingOptions":
        """Convert to legacy ProcessingOptions for the monolith path.

        Generated programmatically to prevent silent drift: any field added to
        ProcessingOptions with a matching PipelineConfig name is auto-mapped.
        Fields with different names are handled in `renames`; fields with no
        PipelineConfig counterpart fall through to their ProcessingOptions default.
        """
        import dataclasses as _dc
        from .workers import ProcessingOptions

        # ProcessingOptions fields whose names differ from PipelineConfig.
        renames = {
            "corner_confidence_threshold": self.corner_confidence,
            "kornia_device": self.device,
        }
        pc_fields = {f.name for f in _dc.fields(self)}
        kwargs: dict = {"output_dir": output_dir}
        for f in _dc.fields(ProcessingOptions):
            if f.name == "output_dir":
                continue
            elif f.name in renames:
                kwargs[f.name] = renames[f.name]
            elif f.name in pc_fields:
                kwargs[f.name] = getattr(self, f.name)
            else:
                raise RuntimeError(
                    f"ProcessingOptions field '{f.name}' has no source in PipelineConfig. "
                    "Add it to PipelineConfig or to the `renames` dict in to_options()."
                )
        return ProcessingOptions(**kwargs)

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
