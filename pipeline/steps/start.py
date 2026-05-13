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

    # Config knobs (with defaults matching ProcessingOptions)
    queue_size: int = 256
    inference_batch_size: int = 16
    corner_confidence_threshold: float = 0.5
    blur_threshold: float = 30.0
    variance_threshold: float = 20.0
    empty_pixel_threshold: float = 0.98
    group_gap_ms: int = 300
    background_frames: int = 30
    background_threshold: float = 15.0
    null_patience_frames: int = 20
    min_track_length: int = 6
    use_kornia: bool = True
    kornia_device: str = "auto"
    triage_keep_percentile: float = 0.05
    rotate_180: bool = True
    tracker_backend: str = "botsort"
    centroid_jump_ratio: float = 0.30
    centroid_jump_frames: int = 3
    foil_threshold: float = 50.0
    enable_foil_aware_fusion: bool = True
    telemetry_scope: str = "canonical"

    # Filled in after storage registration
    video_id: Optional[int] = None


def init_run(
    video_path: str,
    output_dir: str,
    db_path: str,
    detector: str = "docaligner",
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
    out = Path(output_dir)
    frame_dir = out / "frames"
    crops_dir = out / "crops"
    out.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        video_path=str(video_path),
        output_dir=str(output_dir),
        db_path=str(db_path),
        detector=detector,
        config_preset=config_preset,
        frame_dir=str(frame_dir),
        crops_dir=str(crops_dir),
        **kwargs,
    )

    from card_capture.storage import Storage
    from card_capture.pipeline import _file_hash

    storage = Storage(Path(db_path))
    storage.initialize()

    vp = Path(video_path)
    ctx.video_id = storage.add_video(
        source_path=str(vp),
        file_hash=_file_hash(vp) if vp.exists() else "fake",
        duration_ms=0,
        width=0,
        height=0,
    )

    return ctx
