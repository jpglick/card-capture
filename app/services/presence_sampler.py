from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from card_capture.data.connection import open_connection, read_connection

SAMPLES_PER_RUN = 20
_SCAN_FPS = 15.0
_SCAN_WIDTH = 192


def sample_presence_frames(
    video_path: Path,
    run_id: str,
    video_id: int,
    output_dir: Path,
    db_path: Path,
) -> int:
    """Re-scan video at 192px/15fps, save SAMPLES_PER_RUN balanced frames.

    Returns the number of rows inserted into presence_samples.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return 0

    target_present, target_absent = _balance_targets(db_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(source_fps / _SCAN_FPS)))

    scan_frames: list[tuple[int, int, np.ndarray]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            h, w = frame.shape[:2]
            scaled_h = max(1, int(round(h * _SCAN_WIDTH / w)))
            small = cv2.resize(frame, (_SCAN_WIDTH, scaled_h))
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            scan_frames.append((frame_idx, ts_ms, small))
        frame_idx += 1
    cap.release()

    if not scan_frames:
        return 0

    total_target = target_present + target_absent
    if len(scan_frames) <= total_target:
        selected = scan_frames
    else:
        step = len(scan_frames) / total_target
        selected = [scan_frames[int(i * step)] for i in range(total_target)]

    out_dir = Path(output_dir) / "presence_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    inserted = 0
    with open_connection(db_path) as conn:
        for fi, ts_ms, small in selected:
            fname = f"{run_id}_{fi}.jpg"
            fpath = out_dir / fname
            cv2.imwrite(str(fpath), small)
            conn.execute(
                """INSERT INTO presence_samples
                   (run_id, video_id, frame_index, timestamp_ms, image_path)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, video_id, fi, ts_ms, str(fpath)),
            )
            inserted += 1
    return inserted


def _balance_targets(db_path: Path) -> tuple[int, int]:
    half = SAMPLES_PER_RUN // 2
    try:
        with read_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT label, COUNT(*) FROM presence_samples "
                "WHERE label IS NOT NULL GROUP BY label"
            ).fetchall()
    except Exception:
        return half, half

    counts = dict(rows)
    present = counts.get("present", 0)
    absent = counts.get("absent", 0)

    if present > absent * 3:
        return SAMPLES_PER_RUN // 4, 3 * SAMPLES_PER_RUN // 4
    if absent > present * 3:
        return 3 * SAMPLES_PER_RUN // 4, SAMPLES_PER_RUN // 4
    return half, half
