"""Step 0: Initialise a pipeline run.

``init_run`` creates the output directories, opens the SQLite database,
registers the video, and returns a ``RunContext`` that every subsequent
step receives as its first argument.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunContext:
    """Serialisable configuration for a single pipeline run.

    All fields are plain Python scalars or strings so that Metaflow can
    pickle the object as a step artifact without trouble.
    """

    video_path: str
    output_dir: str
    db_path: str
    detector: str
    config_preset: str

    # Derived paths (set by init_run)
    frame_dir: str = ""
    crops_dir: str = ""

    # Config knobs (aligned with card_capture.config.PipelineConfig)
    reader_backend: str = "auto"
    queue_size: int = 256
    inference_batch_size: int = 16
    device: str = "auto"
    
    corner_confidence: float = 0.5
    blur_threshold: float = 30.0
    variance_threshold: float = 20.0
    empty_pixel_threshold: float = 0.98
    detection_width: int = 640
    
    background_frames: int = 30
    background_threshold: float = 15.0
    presence_threshold: float = 0.4
    null_patience_frames: int = 30
    triage_keep_percentile: float = 0.05
    fast_scan_fps: float = 15.0
    confirm_scan_fps: float = 5.0
    valley_drop_ratio: float = 0.40
    valley_min_width_frames: int = 3
    delta_spike_ratio: float = 0.50
    
    tracker_backend: str = "bytetrack"
    group_gap_ms: int = 300
    spatial_variance_threshold: float = 300.0
    min_track_length: int = 3
    centroid_jump_ratio: float = 0.30
    centroid_jump_frames: int = 3
    
    rotate_180: bool = False
    reid_distance_threshold: float = 0.6
    fusion_target_frames: int = 4
    foil_threshold: float = 50.0
    enable_foil_aware_fusion: bool = True
    corner_refinement: bool = False
    
    use_kornia: bool = True
    kornia_device: str = "auto"
    
    use_fb_classifier: bool = True
    use_dino_dedup: bool = True
    telemetry_scope: str = "canonical"

    # Per-video adaptive distributions (collected during run)
    observed_novelty_scores: list[float] = field(default_factory=list)
    observed_intra_track_distances: list[float] = field(default_factory=list)

    # Filled in after storage registration
    video_id: Optional[int] = None


def init_run(
    video_path: str,
    output_dir: str,
    db_path: str,
    detector: Optional[str] = None,
    config_preset: str = "balanced",
    **kwargs,
) -> RunContext:
    """Create output directories, initialise the database, and return a RunContext.

    Args:
        video_path:   Absolute path to the source video file.
        output_dir:   Root directory where frames/ and crops/ will be created.
        db_path:      Path to the SQLite database (created if absent).
        detector:     Detector backend key (``"fake"`` or ``"docaligner"``).
        config_preset: Named config preset (``"balanced"``, ``"fast"``, ``"quality"``).
        **kwargs:     Override any ``RunContext`` field by keyword.

    Returns:
        A fully-initialised ``RunContext``.
    """
    from card_capture.config import PipelineConfig
    cfg = PipelineConfig()
    
    # Apply keyword overrides to the config object first
    if detector:
        cfg.detector = detector
        
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    out = Path(output_dir)
    frame_dir = out / "frames"
    crops_dir = out / "crops"
    out.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Convert cfg to RunContext fields
    ctx = RunContext(
        video_path=str(video_path),
        output_dir=str(output_dir),
        db_path=str(db_path),
        detector=cfg.detector,
        config_preset=config_preset,
        frame_dir=str(frame_dir),
        crops_dir=str(crops_dir),
        
        reader_backend=cfg.reader_backend,
        queue_size=cfg.queue_size,
        inference_batch_size=cfg.inference_batch_size,
        device=cfg.device,
        
        corner_confidence=cfg.corner_confidence,
        blur_threshold=cfg.blur_threshold,
        variance_threshold=cfg.variance_threshold,
        empty_pixel_threshold=cfg.empty_pixel_threshold,
        detection_width=cfg.detection_width,
        
        background_frames=cfg.background_frames,
        background_threshold=cfg.background_threshold,
        presence_threshold=cfg.presence_threshold,
        null_patience_frames=cfg.null_patience_frames,
        triage_keep_percentile=cfg.triage_keep_percentile,
        fast_scan_fps=cfg.fast_scan_fps,
        confirm_scan_fps=cfg.confirm_scan_fps,
        valley_drop_ratio=cfg.valley_drop_ratio,
        valley_min_width_frames=cfg.valley_min_width_frames,
        delta_spike_ratio=cfg.delta_spike_ratio,
        
        tracker_backend=cfg.tracker_backend,
        group_gap_ms=cfg.group_gap_ms,
        spatial_variance_threshold=cfg.spatial_variance_threshold,
        min_track_length=cfg.min_track_length,
        centroid_jump_ratio=cfg.centroid_jump_ratio,
        centroid_jump_frames=cfg.centroid_jump_frames,
        
        rotate_180=cfg.rotate_180,
        reid_distance_threshold=cfg.reid_distance_threshold,
        fusion_target_frames=cfg.fusion_target_frames,
        foil_threshold=cfg.foil_threshold,
        enable_foil_aware_fusion=cfg.enable_foil_aware_fusion,
        corner_refinement=cfg.corner_refinement,
        
        use_kornia=cfg.use_kornia,
        kornia_device=cfg.device,
    )

    from card_capture.storage import Storage
    from card_capture.pipeline import _file_hash

    storage = Storage(Path(db_path))
    storage.initialize()

    vp = Path(video_path)
    ctx.video_id = storage.get_or_create_video(
        source_path=str(vp),
        file_hash=_file_hash(vp) if vp.exists() else "fake",
    )

    return ctx
