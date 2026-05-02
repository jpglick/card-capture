from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2

from .cropper import CardCropper
from .detectors import CardDetector
from .models import ProcessingResult
from .scoring import QualityScorer
from .selector import CandidateSelector, ScoredCandidate
from .storage import Storage


@dataclass(frozen=True)
class ProcessingOptions:
    output_dir: Path
    sample_fps: float = 5.0
    max_candidates: int = 10
    confidence_threshold: float = 0.25
    group_gap_ms: int = 1000
    detections_to_stop: int = 1
    quality_floor: float = 0.5


class VideoProcessor:
    def __init__(
        self,
        storage: Storage,
        sampler,
        detector: CardDetector,
        cropper: Optional[CardCropper] = None,
        scorer: Optional[QualityScorer] = None,
        selector: Optional[CandidateSelector] = None,
    ):
        self.storage = storage
        self.sampler = sampler
        self.detector = detector
        self.cropper = cropper or CardCropper()
        self.scorer = scorer or QualityScorer()
        self.selector = selector

    def process(self, video_path: Path, options: ProcessingOptions) -> ProcessingResult:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video does not exist: {video_path}")

        options.output_dir.mkdir(parents=True, exist_ok=True)
        crop_dir = options.output_dir / "crops"
        frame_dir = options.output_dir / "frames"
        best_dir = options.output_dir / "best"
        crop_dir.mkdir(parents=True, exist_ok=True)
        frame_dir.mkdir(parents=True, exist_ok=True)
        best_dir.mkdir(parents=True, exist_ok=True)

        video_id = self.storage.add_video(
            source_path=str(video_path),
            file_hash=_file_hash(video_path),
            duration_ms=0,
            width=0,
            height=0,
        )

        candidates: List[ScoredCandidate] = []
        detection_count = 0
        good_detection_count = 0

        for frame in self.sampler.sample(video_path, options.sample_fps):
            source_frame_path = frame_dir / f"video_{video_id}_frame_{frame.frame_index}.jpg"
            cv2.imwrite(str(source_frame_path), frame.image)

            stop_this_frame = False
            for detection in self.detector.detect(frame):
                if detection.confidence < options.confidence_threshold:
                    continue
                crop = self.cropper.crop(frame.image, detection.polygon)
                score = self.scorer.score(crop.image, detection.confidence)
                crop_path = crop_dir / (
                    f"video_{video_id}_frame_{frame.frame_index}_det_{detection_count}.jpg"
                )
                cv2.imwrite(str(crop_path), crop.image)
                detection_id = self.storage.add_detection(
                    video_id=video_id,
                    detection=detection,
                    crop_path=str(crop_path),
                    source_frame_path=str(source_frame_path),
                    score=score,
                    crop_width=crop.width,
                    crop_height=crop.height,
                )
                candidates.append(
                    ScoredCandidate(
                        detection_id=detection_id,
                        timestamp_ms=frame.timestamp_ms,
                        image_path=str(crop_path),
                        score=score,
                    )
                )
                detection_count += 1
                if (
                    options.detections_to_stop > 0
                    and score.total >= options.quality_floor
                ):
                    good_detection_count += 1
                    if good_detection_count >= options.detections_to_stop:
                        stop_this_frame = True
                        break  # stop processing further detections in this frame

            if stop_this_frame:
                break  # stop consuming more frames

        selector = self.selector or CandidateSelector(
            group_gap_ms=options.group_gap_ms,
            max_candidates=options.max_candidates,
        )
        selected = selector.select(candidates)
        for selected_index, candidate in enumerate(selected):
            source_path = Path(candidate.image_path)
            best_path = best_dir / f"video_{video_id}_best_{selected_index + 1}.jpg"
            shutil.copyfile(source_path, best_path)
            self.storage.add_saved_card(
                detection_id=candidate.detection_id,
                image_path=str(best_path),
                final_score=candidate.score.total,
            )

        self.storage.update_video_status(
            video_id,
            "complete" if selected else "no_detections",
        )
        return ProcessingResult(
            video_id=video_id,
            detection_count=detection_count,
            saved_count=len(selected),
            output_dir=options.output_dir,
        )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
