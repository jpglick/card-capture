from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np
import torch

from .interfaces import CardDetector, FrameSampler
from .models import (
    CardDetection,
    DetectionPacket,
    FramePacket,
    FrameSample,
    ProcessingResult,
    TrackState,
)


@dataclass(frozen=True)
class PipelineRunRequest:
    video_path: Path
    output_dir: Path
    db_path: Path
    video_id: int
    detector_backend: str = "auto"
    config_preset: str = "balanced"
    runtime_mode: str = "strict_gpu"  # strict_gpu | cpu_debug
    fusion_target_frames: int = 1
    corner_refinement: bool = False
    presence_threshold: float = 0.4
    min_track_length: int = 3
    rotate_180: bool = False


@dataclass(frozen=True)
class PipelineRunResult:
    run_id: str
    success: bool
    processing_result: Optional[ProcessingResult] = None
    error: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)


class PipelineRuntime(Protocol):
    def run(self, request: PipelineRunRequest) -> PipelineRunResult:
        ...


class UnifiedRuntime:
    """In-process pipeline execution runtime.
    
    Consolidates sampling, detection, tracking, and refinement into a single
    loop to eliminate process overhead and redundant work (like re-decoding).
    """

    def __init__(
        self,
        sampler: FrameSampler,
        detector: CardDetector,
    ):
        self.sampler = sampler
        self.detector = detector

    def run(self, request: PipelineRunRequest) -> PipelineRunResult:
        from pipeline.steps import (
            detect, novelty, track, refine, score, resolve, fuse, dedup, store
        )
        from pipeline.steps.start import RunContext

        start_time = time.time()
        _run_id = f"run_{int(start_time)}"
        
        try:
            # 0. Setup Context
            ctx = RunContext(
                video_path=str(request.video_path),
                output_dir=str(request.output_dir),
                db_path=str(request.db_path),
                detector=request.detector_backend,
                config_preset=request.config_preset,
                crops_dir=str(request.output_dir / "crops"),
                frame_dir=str(request.output_dir / "frames"),
                rotate_180=request.rotate_180,
                video_id=request.video_id,
                fusion_target_frames=request.fusion_target_frames,
                corner_refinement=request.corner_refinement,
                presence_threshold=request.presence_threshold,
                min_track_length=request.min_track_length,
            )
            Path(ctx.crops_dir).mkdir(parents=True, exist_ok=True)
            Path(ctx.frame_dir).mkdir(parents=True, exist_ok=True)

            # 1. Detection (Fused path if possible, but here we stay in-process)
            crop_cache: Dict[int, np.ndarray] = {}
            _t0 = time.time()
            # Note: We use the existing _run_fused_inference because it already
            # handles the threading producer/consumer model and crop caching.
            detect_out = detect._run_fused_inference(
                ctx, self.sampler, self.detector, 
                Path(ctx.output_dir), Path(ctx.frame_dir), 
                crop_cache=crop_cache
            )
            t_detect = time.time() - _t0
            
            detect._save_corner_samples(ctx, detect_out.detection_rows, Path(ctx.output_dir))

            # 2. Novelty Gating
            _t0 = time.time()
            novelty_out = novelty.run(ctx, detect_out)
            t_novelty = time.time() - _t0

            # 3. Tracking
            _t0 = time.time()
            track_out = track.run(ctx, novelty_out)
            t_track = time.time() - _t0

            # 4. Refinement (consuming cached crops)
            # Prune cache first to save memory
            active_det_ids = {
                cand["detection_id"] 
                for t in track_out.tracks_data 
                for cand in t.get("candidates", [])
            }
            crop_cache = {k: v for k, v in crop_cache.items() if k in active_det_ids}

            _t0 = time.time()
            refine_out = refine.run(ctx, track_out, decoded_crops=crop_cache)
            t_refine = time.time() - _t0

            # 5. Scoring
            _t0 = time.time()
            score_out = score.run(ctx, refine_out)
            t_score = time.time() - _t0

            # 6. Front/Back Resolution
            _t0 = time.time()
            resolve_out = resolve.run(ctx, score_out)
            t_resolve = time.time() - _t0

            # 7. Fusion
            _t0 = time.time()
            fused_canonicals = []
            for prepared_track in resolve_out.prepared_tracks:
                if prepared_track is None: continue
                fuse_out = fuse.run(ctx, prepared_track)
                if fuse_out is not None:
                    fused_canonicals.append(fuse_out.fused_canonical)
            t_fuse = time.time() - _t0

            # 8. Deduplication
            _t0 = time.time()
            dedup_out = dedup.run(ctx, fused_canonicals)
            t_dedup = time.time() - _t0

            # 9. Storage
            _t0 = time.time()
            store_out = store.run(
                ctx,
                dedup_out.dedup_groups,
                fused_canonicals,
                prepared_tracks=resolve_out.prepared_tracks,
                run_id=_run_id
            )
            t_store = time.time() - _t0

            total_elapsed = time.time() - start_time
            
            res = ProcessingResult(
                video_id=request.video_id,
                frame_count=detect_out.frame_count,
                accepted_frame_count=detect_out.accepted_frame_count,
                detection_count=len(detect_out.detection_rows),
                saved_instance_count=len(fused_canonicals),
                output_dir=request.output_dir,
                telemetry={
                    "t_detect": t_detect,
                    "t_novelty": t_novelty,
                    "t_track": t_track,
                    "t_refine": t_refine,
                    "t_score": t_score,
                    "t_resolve": t_resolve,
                    "t_fuse": t_fuse,
                    "t_dedup": t_dedup,
                    "t_store": t_store,
                    "total_elapsed": total_elapsed,
                }
            )

            return PipelineRunResult(
                run_id=_run_id,
                success=True,
                processing_result=res,
                telemetry=res.telemetry,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return PipelineRunResult(
                run_id=_run_id,
                success=False,
                error=str(e),
                telemetry={"total_elapsed_s": time.time() - start_time},
            )
