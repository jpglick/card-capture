"""Step 1 — detect: run Stages 1–3 (sampler + triage + YOLO-OBB detection).

Wraps the existing ``_run_pipeline_workers`` producer/consumer subsystem.
Returns serialisable lists so that Metaflow can pickle the artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json as _json
import sqlite3 as _sqlite3

from pipeline.steps.start import RunContext
from card_capture.gpu_refinement import KorniaNormalizer


@dataclass
class DetectOutput:
    """Outputs of the detect step — all JSON-serialisable."""

    frame_count: int
    accepted_frame_count: int
    # (frame_index, timestamp_ms, is_present) tuples
    accepted_frame_presence: List[Tuple[int, int, bool]]
    # Flat list of detection dicts (one per CornerDetection)
    detection_rows: List[Dict[str, Any]]
    # Sampler telemetry dict (all values must be JSON-serialisable)
    sampler_telemetry: Dict[str, Any]
    video_id: int
    # Diagnostic snapshot for the detect stage (frames, YOLO device/timing, triage rate)
    detect_telemetry: Dict[str, Any] = field(default_factory=dict)


def run(ctx: RunContext) -> DetectOutput:
    """Execute Stages 1–3 and return serialised detection data.
    """
    from pathlib import Path as _Path
    from card_capture.workers import (
        ProcessingOptions,
        _run_pipeline_workers,
    )

    video_path = _Path(ctx.video_path)
    output_dir = _Path(ctx.output_dir)
    frame_dir = _Path(ctx.frame_dir)

    sampler, detector = _build_sampler_detector(ctx)

    options = _ctx_to_options(ctx, output_dir)

    if ctx.detector in ("cuda", "mps", "docaligner"):
        # High-performance path: prefetch decode + batch YOLO + eager warp
        detect_out = _run_fused_inference(ctx, sampler, detector, output_dir, frame_dir)
        _save_corner_samples(ctx, detect_out.detection_rows, output_dir)
        return detect_out

    stats, consumer_stats, raw_rows = _run_pipeline_workers(
        video_path=video_path,
        video_id=ctx.video_id,
        frame_dir=frame_dir,
        sampler=sampler,
        detector=detector,
        options=options,
    )

    # Serialise _DetectionEnvelope → plain dict
    detection_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        dp = row.detection_packet
        cd = dp.corner_detection
        detection_rows.append(
            {
                "detection_id": idx,
                "frame_index": dp.frame_index,
                "timestamp_ms": dp.timestamp_ms,
                "width": dp.width,
                "height": dp.height,
                "corners": [(float(p[0]), float(p[1])) for p in cd.corners],
                "confidence": float(cd.confidence),
                "source_frame_path": row.source_frame_path,
                "triage_metrics": dict(row.triage_metrics),
            }
        )

    # Make sampler_telemetry JSON-serialisable
    sampler_telemetry = _serialise_telemetry(stats.sampler_telemetry)
    sampler_telemetry.setdefault("target_yolo_fps", ctx.target_yolo_fps)

    _save_corner_samples(ctx, detection_rows, output_dir)

    triage_pass_rate = (
        stats.accepted_frame_count / stats.frame_count
        if stats.frame_count > 0 else 0.0
    )
    detect_telemetry: Dict[str, Any] = {
        "frame_count": stats.frame_count,
        "accepted_frame_count": stats.accepted_frame_count,
        "triage_pass_rate": round(triage_pass_rate, 4),
        "yolo_frames": consumer_stats.yolo_frames,
        "yolo_batches": consumer_stats.yolo_batches,
        "yolo_elapsed_s": round(consumer_stats.yolo_elapsed_s, 2),
        "yolo_device": consumer_stats.device_resolved,
        "presence_windows": stats.sampler_telemetry.get("last_presence_window_count"),
        "sampler_type": stats.sampler_telemetry.get("sampler_type"),
    }

    return DetectOutput(
        frame_count=stats.frame_count,
        accepted_frame_count=stats.accepted_frame_count,
        accepted_frame_presence=list(stats.accepted_frame_presence),
        detection_rows=detection_rows,
        sampler_telemetry=sampler_telemetry,
        video_id=ctx.video_id,
        detect_telemetry=detect_telemetry,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_corner_samples(ctx: RunContext, detection_rows: list, output_dir: Path) -> None:
    """Persist borderline YOLO detections (0.50–0.70 conf) for corner labeling."""
    if not ctx.db_path:
        return
    borderline = [d for d in detection_rows if 0.50 <= d["confidence"] <= 0.70]
    if not borderline:
        return
    try:
        with _sqlite3.connect(ctx.db_path) as conn:
            for d in borderline:
                conn.execute(
                    """INSERT OR IGNORE INTO corner_samples
                       (run_id, video_id, frame_index, image_path,
                        predicted_corners, confidence)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        getattr(ctx, 'ui_run_id', None) or "",
                        ctx.video_id or 0,
                        d["frame_index"],
                        d.get("source_frame_path", ""),
                        _json.dumps(d["corners"]),
                        d["confidence"],
                    ),
                )
            conn.commit()
    except Exception as exc:
        print(f"[detect] corner sampling failed: {exc}", flush=True)


def _annotate_cheap_scores(row: dict, frame_bgr, device: str = "auto") -> None:
    """Attach warp-free flatness + clarity to a detection row's triage_metrics.

    `frame_bgr` is the full-res HxWxC BGR frame (numpy). Corners are in
    full-res pixel coords. Computed before any Kornia warp so it is cheap.
    """
    from card_capture.frame_quality import flatness_score, clarity_score_gpu
    corners = row.get("corners") or []
    flatness = flatness_score(corners)
    clarity = 0.0
    if corners:
        xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        clarity = clarity_score_gpu(frame_bgr, bbox, device=device)
    tm = row.setdefault("triage_metrics", {})
    tm["flatness"] = round(float(flatness), 6)
    tm["clarity"] = round(float(clarity), 6)


def _cheap_metrics_cpu(small_bgr, bbox):
    """Warp-free clarity (Laplacian variance) + 8x8 average-hash of a bbox ROI,
    computed on the CPU from the small (640px) frame.

    The detect pass used to slice these ROIs out of the full-res 4K frame on the
    GPU, but each detection fired ~10 tiny MPS kernels whose per-launch overhead
    dominated (~19ms/det, ~45% of the detect stage). Computing on the small CPU
    frame is ~16x faster and — because clarity is only used to *rank* a track's
    candidates — preserves the ranking (Spearman ~0.95 vs the full-res score).
    Returns (clarity_variance, ahash_int).
    """
    import cv2
    from card_capture.frame_quality import ahash_from_grid
    h, w = small_bgr.shape[:2]
    x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    x0 = max(0, min(x0, w - 1)); x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(y0 + 1, min(y1, h))
    roi = small_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Scale to match the legacy GPU clarity (which ran on [0,1] luma * 255**2).
    clarity = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    grid = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype("float32").ravel() / 255.0
    return clarity, ahash_from_grid(grid)


def _build_sampler_detector(ctx: RunContext):
    """Construct the sampler and detector from RunContext.detector."""
    import platform
    from pathlib import Path as _Path
    from card_capture.detectors import FakeCardDetector, CardcaptorUltralyticsDetector
    from card_capture.sampler import SyntheticSampler, AdaptivePresenceSampler, StrideSampler

    if ctx.detector == "fake":
        sampler = SyntheticSampler()
        detector = FakeCardDetector()
    elif ctx.detector == "cuda":
        from card_capture.sampler.cuda_sampler import CudaSampler
        sampler = CudaSampler(
            video_path=_Path(ctx.video_path),
            stride=ctx.cuda_stride,
            opening_scan_s=ctx.opening_scan_s,
        )
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=ctx.corner_confidence,
            detection_width=640,
            device="cuda",   # explicit — hard-fail if CUDA unavailable on instance
        )
    else:
        # High-res local sampling: default to 15 fps for MPS/CPU runs to 
        # improve tracking stability on the Mac Mini.
        local_fps = 15.0 if ctx.target_yolo_fps == 3.0 else ctx.target_yolo_fps
        
        # Use NV12 if we are on a Mac with a local GPU detector to bypass slow CPU color conversion.
        # This is requested for both 'mps' and 'docaligner' (which uses the local GPU).
        use_nv12 = ctx.detector in ("mps", "docaligner")
        
        # Prefer OpenCV on Mac: it uses AVFoundation natively, perfectly handling 
        # Apple's 4K HEVC streams via the Media Engine. PyAV/FFmpeg chokes on them.
        reader_backend = "auto"
        if platform.system() == "Darwin":
            reader_backend = "opencv"

        sampler = StrideSampler(
            video_path=_Path(ctx.video_path),
            target_yolo_fps=local_fps,
            reader_backend=reader_backend,
            pixel_format="bgr24",
        )
        detector = CardcaptorUltralyticsDetector(
            confidence_threshold=ctx.corner_confidence,
            detection_width=640,
            device="auto",
        )
    return sampler, detector


def _run_fused_inference(
    ctx: RunContext,
    sampler: Union["CudaSampler", "StrideSampler"],
    detector: "CardcaptorUltralyticsDetector",
    output_dir: Path,
    frame_dir: Path,
    crop_cache: Optional[dict] = None,
) -> "DetectOutput":
    """High-Performance 4K Pipeline: Full Background Parallelization.

    Achieves >60 FPS on M4 by moving ALL heavy work to worker threads:
    1. Decoder thread yields raw frames.
    2. GPU Worker thread does zero-copy upload, YOLO inference, and GPU metrics.
    3. Main thread handles ONLY tracking (negligible CPU cost).
    """
    from card_capture.models import FramePacket
    from card_capture.ml import gpu_ops
    from card_capture.ml.gpu_ops import gpu_roi_mean_abs_diff
    from card_capture.presence.background_novelty import BackgroundModel
    from card_capture.frame_quality import (
        flatness_score, clarity_var_gpu_roi, appearance_grid_gpu_roi, ahash_from_grid
    )
    import numpy as np
    import torch
    import torch.nn.functional as F
    import cv2
    import time as _time
    import queue as _queue
    import threading as _threading

    detection_rows: list[dict] = []
    accepted_frame_presence: list[tuple[int, int, bool]] = []
    det_id = 0
    total_frames = 0
    yolo_elapsed_s = 0.0
    yolo_batches = 0

    from card_capture.detectors import probe_torch_device_status
    from card_capture import gpu_utils
    if ctx.kornia_device == "mps":
        resolved_device = gpu_utils.require_device("mps").type
    else:
        resolved_device = probe_torch_device_status(ctx.kornia_device).resolved

    normalizer = None
    if crop_cache is not None:
        try:
            normalizer = KorniaNormalizer(width=750, height=1050, device=resolved_device)
        except Exception as e:
            print(f"[detect] KorniaNormalizer unavailable: {e}", flush=True)
            crop_cache = None

    # NOVELTY SUPPORT: Prepare GPU-resident background model
    bg_images = getattr(sampler, "background_proxies", [])
    bg_model = BackgroundModel.from_frames(bg_images) if bg_images else None
    bg_t = None
    if bg_model is not None:
        bg_small = cv2.resize(bg_model.mean_bgr, (640, 640))
        bg_t = torch.from_numpy(bg_small).to(resolved_device).permute(2, 0, 1).float() / 255.0

    # Pipeline stages:
    _q_raw: "_queue.Queue" = _queue.Queue(maxsize=16)
    _q_fused: "_queue.Queue" = _queue.Queue(maxsize=16)
    _SENTINEL = object()

    batch_size = 4 if resolved_device == "mps" else 8
    
    from card_capture.ingestion import probe_video
    _meta = probe_video(ctx.video_path)
    _source_fps = _meta.get("fps", 30.0) or 30.0
    _total_vframes = _meta.get("frame_count", 1) or 1
    _target_fps = ctx.target_yolo_fps
    _frame_step = max(1, int(round(_source_fps / _target_fps))) if _target_fps > 0 else 1
    total_expected = max(1, _total_vframes // _frame_step)

    def _producer():
        """Stage 1: Decode video frames and CPU resize."""
        try:
            batch = []
            frames_decoded = 0
            t_last = _time.time()
            for frame in sampler.sample():
                # OpenCV reuses internal buffers. We MUST copy the array so the 
                # background thread doesn't overwrite it while it sits in the queue.
                safe_image = frame.image.copy()
                frame.image = safe_image
                
                # CPU Squeeze is ~5ms. 10x faster than GPU float interpolation.
                small_img = cv2.resize(safe_image, (640, 640))
                batch.append((frame, small_img))
                
                frames_decoded += 1
                if frames_decoded % 20 == 0:
                    pct = int(frames_decoded * 100 / total_expected)
                    t_now = _time.time()
                    fps = 20 / (t_now - t_last) if (t_now - t_last) > 0 else 0
                    t_last = t_now
                    print(f"[progress] detect.decoder {pct}% detail: Decoded {frames_decoded}/{total_expected} ({fps:.1f} fps)", flush=True)

                if len(batch) >= batch_size:
                    _q_raw.put(batch)
                    batch = []
            if batch:
                _q_raw.put(batch)
            
            # Final 100%
            print(f"[progress] detect.decoder 100% detail: Decoded {frames_decoded}/{total_expected}", flush=True)
        except Exception as e:
            _q_raw.put(("__error__", e))
        finally:
            _q_raw.put(_SENTINEL)

    def _gpu_worker():
        """Stage 2: GPU-accelerated prep + YOLO + Metrics."""
        nonlocal det_id, yolo_elapsed_s, yolo_batches
        frames_prepped = 0
        t_last = _time.time()
        while True:
            batch_tuples = _q_raw.get()
            if batch_tuples is _SENTINEL:
                _q_fused.put(_SENTINEL)
                # Final 100%
                print(f"[progress] detect.gpu 100% detail: Prepped {frames_prepped}/{total_expected}", flush=True)
                break
            if isinstance(batch_tuples, tuple) and batch_tuples[0] == "__error__":
                _q_fused.put(batch_tuples)
                break
            
            try:
                # 1. Upload only the small (640) frames for YOLO + novelty. The
                # full-res 4K frames stay on the CPU: cheap metrics (clarity/ahash)
                # are now computed from the small frame on the CPU (see
                # _cheap_metrics_cpu), so there is no reason to push ~25MB/frame
                # across to MPS — that upload plus the per-detection GPU metric
                # kernels were ~half of the detect stage's wall time.
                tensors_small = []
                frames = []
                small_imgs = []
                tensors_full = []
                for f, f_small in batch_tuples:
                    frames.append(f)
                    small_imgs.append(f_small)
                    tensors_small.append(torch.from_numpy(f_small).to(resolved_device))
                    # The full-res 4K frame is only needed on-device for the GPU
                    # warp (fused CUDA path). The MPS path warps later in refine,
                    # so it never uploads 4K here.
                    if crop_cache is not None:
                        tensors_full.append(torch.from_numpy(f.image).to(resolved_device))

                small_bgr_t = torch.stack(tensors_small)
                full_res_bgr_t = torch.stack(tensors_full) if tensors_full else None
                
                # 2. YOLO Prep (small tensor only)
                batch_rgb_t = small_bgr_t[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2).float() / 255.0
                yolo_input_t = batch_rgb_t  # already 640x640
                
                if resolved_device == "mps":
                    # THE MAGIC SYNC: Without this, CoreML stalls for ~100ms per frame
                    # resolving PyTorch's async memory pointers before Neural Engine inference!
                    torch.mps.synchronize()
                
                # 3. YOLO Inference
                _t_start = _time.time()
                packets_out = detector.detect_batch(frames, detector.confidence_threshold, tensor_input=yolo_input_t)
                yolo_elapsed_s += _time.time() - _t_start
                yolo_batches += 1
                
                # 4. Metrics & Warping. Clarity + ahash are computed on the CPU from
                # the small (640) frame — see _cheap_metrics_cpu. Novelty stays on the
                # GPU (it already only needs the small tensor). Full-res frames are
                # touched only for the warp (fused CUDA path).
                rows = []
                _gpu_tasks = []
                _warp_tasks = []
                for pkt in packets_out:
                    cd = pkt.corner_detection

                    idx = -1
                    for i, f in enumerate(frames):
                        if f.frame_index == pkt.frame_index:
                            idx = i; break
                    if idx == -1: continue

                    # bbox in 640 space (for novelty + CPU metrics)
                    sx, sy = 640 / pkt.width, 640 / pkt.height
                    c640 = [(p[0] * sx, p[1] * sy) for p in cd.corners]
                    xs6 = [p[0] for p in c640]; ys6 = [p[1] for p in c640]
                    bbox_640 = (int(min(xs6)), int(min(ys6)), int(max(xs6)), int(max(ys6)))

                    novelty_t = torch.tensor(1.0, device=resolved_device)
                    if bg_t is not None:
                        frame_small_bgr = yolo_input_t[idx][[2, 1, 0], :, :]
                        novelty_t = gpu_roi_mean_abs_diff(frame_small_bgr, bg_t, bbox_640)

                    # Cheap clarity + ahash on the CPU small frame (~16x faster than
                    # slicing the 4K ROI on MPS, ranking preserved).
                    clarity, ahash = _cheap_metrics_cpu(small_imgs[idx], bbox_640)

                    row = {
                        "frame_index": pkt.frame_index,
                        "timestamp_ms": pkt.timestamp_ms,
                        "width": pkt.width,
                        "height": pkt.height,
                        "corners": [(float(p[0]), float(p[1])) for p in cd.corners],
                        "confidence": float(cd.confidence),
                        "triage_metrics": {
                            "flatness": round(float(flatness_score(cd.corners)), 6),
                            "clarity": round(float(clarity), 6),
                            "ahash": ahash,
                        },
                        "novelty_score": 1.0,
                    }
                    _gpu_tasks.append((row, novelty_t))

                    if crop_cache is not None and full_res_bgr_t is not None:
                        # BBox Pre-Crop: Drastically speeds up Kornia warp (100ms -> 2ms)
                        corners = row["corners"]
                        xs_full = [p[0] for p in corners]; ys_full = [p[1] for p in corners]
                        frame_full = full_res_bgr_t[idx]
                        pad = 10
                        c_x0 = max(0, int(min(xs_full)) - pad)
                        c_y0 = max(0, int(min(ys_full)) - pad)
                        c_x1 = min(w, int(max(xs_full)) + pad)
                        c_y1 = min(h, int(max(ys_full)) + pad)

                        frame_crop = frame_full[c_y0:c_y1, c_x0:c_x1]
                        corners_crop = [(p[0] - c_x0, p[1] - c_y0) for p in cd.corners]

                        _warp_tasks.append((frame_crop, corners_crop, row))

                    rows.append(row)

                # 6. Eager Warp (Batched!)
                if crop_cache is not None and _warp_tasks:
                    batch_data = [(fw, corners) for fw, corners, _ in _warp_tasks]
                    try:
                        warped_batch = normalizer.warp_canonical_batch_gpu(
                            batch_data, rotate_180=ctx.rotate_180, return_gpu=True
                        )
                        for i, (_, _, row) in enumerate(_warp_tasks):
                            row["_warped_img"] = warped_batch[i]
                    except Exception as e:
                        print(f"[detect] Batched warp failed: {e}", flush=True)

                if _gpu_tasks:
                    # Single sync to pull all novelty scores off the GPU (clarity +
                    # ahash were already computed on the CPU above).
                    _nov_all = torch.stack([t[1] for t in _gpu_tasks]).cpu().numpy()
                    for i, (row, _) in enumerate(_gpu_tasks):
                        row["novelty_score"] = float(_nov_all[i])

                # Clear frames from CPU memory
                for f in frames: f.image = None
                
                frames_prepped += len(frames)
                if frames_prepped % 20 == 0:
                    pct = int(frames_prepped * 100 / total_expected)
                    t_now = _time.time()
                    fps = 20 / (t_now - t_last) if (t_now - t_last) > 0 else 0
                    t_last = t_now
                    print(f"[progress] detect.gpu {pct}% detail: Prepped {frames_prepped}/{total_expected} ({fps:.1f} fps)", flush=True)

                _q_fused.put((rows, frames))
                del small_bgr_t, yolo_input_t, batch_rgb_t, full_res_bgr_t
                if yolo_batches % 10 == 0 and resolved_device == "mps":
                    torch.mps.empty_cache()
            except Exception as e:
                _q_fused.put(("__error__", e))
                break

    _threading.Thread(target=_producer, daemon=True).start()
    _threading.Thread(target=_gpu_worker, daemon=True).start()

    t_last_main = _time.time()
    while True:
        item = _q_fused.get()
        if item is _SENTINEL: break
        if isinstance(item, tuple) and item[0] == "__error__": raise item[1]
        
        rows, frames = item
        total_frames += len(frames)
        
        # Tracking & ID Assignment
        for row in rows:
            row["detection_id"] = det_id
            if "_warped_img" in row:
                if crop_cache is not None:
                    crop_cache[det_id] = row["_warped_img"]  # ZERO-DOWNLOAD: Keep as GPU tensor
                del row["_warped_img"]
            
            detection_rows.append(row)
            det_id += 1

        for f in frames:
            accepted_frame_presence.append((f.frame_index, f.timestamp_ms, True))

        if total_frames % 20 == 0 or total_frames == len(frames):
            pct = int(total_frames * 100 / total_expected) if total_expected else 0
            t_now = _time.time()
            fps = 20 / (t_now - t_last_main) if (t_now - t_last_main) > 0 else 0
            t_last_main = t_now
            print(f"[progress] detect.main {pct}% detail: Tracked {total_frames}/{total_expected or '?'} ({fps:.1f} fps)", flush=True)

    detect_telemetry = {
        "frame_count": total_frames,
        "accepted_frame_count": total_frames,
        "yolo_frames": total_frames,
        "yolo_batches": yolo_batches,
        "yolo_elapsed_s": round(yolo_elapsed_s, 3),
        "device_resolved": resolved_device,
    }

    return DetectOutput(
        frame_count=total_frames,
        accepted_frame_count=total_frames,
        accepted_frame_presence=accepted_frame_presence,
        detection_rows=detection_rows,
        sampler_telemetry={
            "sampler_type": type(sampler).__name__,
            "last_selected_frame_count": sampler.last_selected_frame_count,
            "last_source_fps": sampler.last_source_fps,
            "last_inter_window_gaps_frames": getattr(sampler, "last_inter_window_gaps_frames", []),
            "target_yolo_fps": getattr(sampler, "target_yolo_fps", None),
            "background_proxies": [p for p in getattr(sampler, "background_proxies", [])],
        },
        video_id=ctx.video_id,
        detect_telemetry=detect_telemetry,
    )


def _ctx_to_options(ctx: RunContext, output_dir: Path):
    """Build a ``ProcessingOptions`` from a ``RunContext``."""
    from card_capture.workers import ProcessingOptions
    return ProcessingOptions(
        output_dir=output_dir,
        queue_size=ctx.queue_size,
        inference_batch_size=ctx.inference_batch_size,
        corner_confidence_threshold=ctx.corner_confidence,
        blur_threshold=ctx.blur_threshold,
        variance_threshold=ctx.variance_threshold,
        empty_pixel_threshold=ctx.empty_pixel_threshold,
        group_gap_ms=ctx.group_gap_ms,
        background_frames=ctx.background_frames,
        background_threshold=ctx.background_threshold,
        null_patience_frames=ctx.null_patience_frames,
        min_track_length=ctx.min_track_length,
        use_kornia=ctx.use_kornia,
        kornia_device=ctx.kornia_device,
        triage_keep_percentile=ctx.triage_keep_percentile,
        rotate_180=ctx.rotate_180,
        tracker_backend=ctx.tracker_backend,
        centroid_jump_ratio=ctx.centroid_jump_ratio,
        centroid_jump_frames=ctx.centroid_jump_frames,
        foil_threshold=ctx.foil_threshold,
        enable_foil_aware_fusion=ctx.enable_foil_aware_fusion,
    )


def _serialise_telemetry(telemetry: dict) -> dict:
    """Convert sampler_telemetry values to JSON-serialisable types."""
    import numpy as np

    result: dict = {}
    for k, v in telemetry.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = float(v)
        elif isinstance(v, list):
            result[k] = [
                int(x) if isinstance(x, np.integer) else
                float(x) if isinstance(x, np.floating) else x
                for x in v
            ]
        else:
            result[k] = v
    return result
