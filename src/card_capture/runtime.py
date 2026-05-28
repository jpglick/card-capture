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


import queue
import threading
from .detectors import probe_torch_device_status

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
            novelty, track, refine, score, resolve, fuse, dedup, store
        )
        from pipeline.steps.start import RunContext
        from pipeline.steps.detect import DetectOutput, _save_corner_samples

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

            # 1. Unified Detection & Sampling Loop
            # We implement the producer/consumer pattern here for strict boundary enforcement
            q_raw = queue.Queue(maxsize=16)
            q_results = queue.Queue()
            crop_cache: Dict[int, np.ndarray] = {}
            
            # Use same device logic as detect.py
            device_status = probe_torch_device_status(request.detector_backend)
            resolved_device = device_status.resolved

            def _producer():
                try:
                    for frame in self.sampler.sample():
                        q_raw.put(frame)
                    q_raw.put(None) # Sentinel
                except Exception as e:
                    q_results.put(("__error__", e))

            def _worker():
                from card_capture.gpu_refinement import KorniaNormalizer
                normalizer = KorniaNormalizer(device=resolved_device)
                
                try:
                    # Batch processing logic
                    while True:
                        batch = []
                        frame = None
                        while len(batch) < 8:
                            try:
                                # First element in batch: block until available
                                # Subsequent elements: don't block too long
                                timeout = None if not batch else 0.01
                                frame = q_raw.get(timeout=timeout)
                                if frame is None: break
                                batch.append(frame)
                            except queue.Empty:
                                break
                        
                        if not batch:
                            if frame is None: break
                            continue
                            
                        # Run detection
                        packets = [
                            FramePacket(
                                frame_index=f.frame_index,
                                timestamp_ms=f.timestamp_ms,
                                image=f.image,
                                width=f.width,
                                height=f.height,
                                triage_metrics={}
                            ) for f in batch
                        ]
                        
                        # Upload to GPU for detection if possible
                        # (Simplification: detect_batch handles upload for now)
                        detections = self.detector.detect_batch(packets, request.presence_threshold)
                        
                        # Eager Warp for crop_cache
                        for pkt in detections:
                            q_results.put(pkt)
                            
                            # Find source frame in batch
                            source_f = next(f for f in batch if f.frame_index == pkt.frame_index)
                            
                            # Warp and cache
                            # (Simplification: Use CPU warp for now to match old detect.py behavior
                            # unless we implement full GPU warp here)
                            # Actually, old detect.py did:
                            # tensors_full.append(torch.from_numpy(f.image).to(resolved_device))
                            # then used Kornia normalizer.
                            
                            # For now, let's just use the existing logic shape:
                            if pkt.corner_detection.confidence >= request.presence_threshold:
                                from card_capture.cropper import CardCropper
                                cropper = CardCropper()
                                crop_res = cropper.crop(source_f.image, pkt.corner_detection.corners)
                                # Assign a stable ID for the cache (we'll use index for now)
                                det_id = pkt.frame_index # Placeholder ID
                                crop_cache[det_id] = crop_res.image
                                # Update pkt with detection_id
                                # (pkt is a DetectionPacket, it doesn't have detection_id field yet
                                # in the model, but the row dict does)
                            
                        if frame is None: break
                        
                    q_results.put(None) # Sentinel
                except Exception as e:
                    q_results.put(("__error__", e))

            prod_thread = threading.Thread(target=_producer, name="producer")
            work_thread = threading.Thread(target=_worker, name="worker")
            
            prod_thread.start()
            work_thread.start()

            detection_packets = []
            while True:
                item = q_results.get()
                if item is None: break
                if isinstance(item, tuple) and item[0] == "__error__": raise item[1]
                detection_packets.append(item)
            
            prod_thread.join()
            work_thread.join()

            # Map to old DetectOutput shape for downstream steps
            rows = []
            for i, p in enumerate(detection_packets):
                det_id = i + 1
                rows.append({
                    "detection_id": det_id,
                    "frame_index": p.frame_index,
                    "timestamp_ms": p.timestamp_ms,
                    "width": p.width,
                    "height": p.height,
                    "corners": p.corner_detection.corners,
                    "confidence": p.corner_detection.confidence,
                    "triage_metrics": {"ahash": 0},
                })
                # Re-map crop_cache to use the new det_id
                if p.frame_index in crop_cache:
                    crop_cache[det_id] = crop_cache.pop(p.frame_index)

            sampler_telemetry = {
                "last_selected_frame_count": getattr(self.sampler, "last_selected_frame_count", 0),
                "last_source_fps": getattr(self.sampler, "last_source_fps", 30.0),
                "last_scan_frame_count": getattr(self.sampler, "last_scan_frame_count", 0),
                "last_inter_window_gaps_frames": getattr(self.sampler, "last_inter_window_gaps_frames", []),
                "last_valley_splits": getattr(self.sampler, "last_valley_splits", []),
            }

            detect_out = DetectOutput(
                frame_count=sampler_telemetry["last_scan_frame_count"],
                accepted_frame_count=sampler_telemetry["last_selected_frame_count"],
                accepted_frame_presence=[], # TODO
                detection_rows=rows,
                sampler_telemetry=sampler_telemetry,
                video_id=request.video_id
            )
            
            t_detect = time.time() - start_time
            
            # ... (Rest of the stages same as before for now)

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
