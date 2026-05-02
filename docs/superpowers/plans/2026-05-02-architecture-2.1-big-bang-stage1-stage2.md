# Architecture 2.1 Stage 1+2 Big-Bang Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current synchronous OpenCV+YOLO pipeline with a multiprocessing Decord/PyAV ingestion plus ONNX-first corner detection pipeline, persisted to the v2.1 schema.

**Architecture:** The runtime is a producer/consumer multiprocessing pipeline. Producer performs frame ingest + triage; consumer performs batch corner detection; parent orchestrates lifecycle and persistence. Storage and CLI are replaced to match v2.1 contracts and breaking schema decisions.

**Tech Stack:** Python 3.9+, `multiprocessing`, `numpy`, `opencv-python`, `decord`, `av` (PyAV), `onnxruntime`, `pytest`, SQLite.

---

## Scope Check

This plan covers one subsystem: Stage 1+2 pipeline replacement plus required schema/CLI/test changes. Stage 3 tracking and Stage 4/5 scoring/rectification are explicitly deferred but schema-ready.

## File Structure and Responsibilities

- Create: `src/card_capture/ingestion.py` - frame readers and triage filter.
- Modify: `src/card_capture/models.py` - v2.1 packet/domain dataclasses.
- Modify: `src/card_capture/detectors.py` - corner detector protocol + ONNX/fake adapters.
- Modify: `src/card_capture/storage.py` - v2.1 schema and persistence API.
- Modify: `src/card_capture/pipeline.py` - multiprocessing orchestration.
- Modify: `src/card_capture/cli.py` - new v2.1 process flags and wiring.
- Modify: `pyproject.toml` - add runtime optional dependencies for Stage 1/2.
- Modify: `README.md` - update process command examples and architecture notes.
- Modify: `QUICK_REFERENCE.md` - align component map with v2.1 architecture.
- Create: `tests/test_models_v21.py` - packet contract tests.
- Create: `tests/test_ingestion.py` - reader backend and triage tests.
- Modify: `tests/test_detectors.py` - corner detector adapter tests.
- Modify: `tests/test_storage.py` - v2.1 schema persistence tests.
- Modify: `tests/test_pipeline.py` - multiprocessing behavior tests.
- Modify: `tests/test_cli.py` - process flags and pipeline wiring tests.

### Task 1: Define v2.1 Models and Pipeline Options

**Files:**
- Modify: `src/card_capture/models.py`
- Modify: `src/card_capture/pipeline.py`
- Test: `tests/test_models_v21.py`

- [ ] **Step 1: Write the failing tests for packet and result contracts**

```python
# tests/test_models_v21.py
from pathlib import Path
import numpy as np
from card_capture.models import FramePacket, CornerDetection, DetectionPacket, ProcessingResult

def test_frame_packet_carries_triage_metrics():
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    packet = FramePacket(3, 120, image, 12, 10, {"blur": 10.0, "variance": 40.0, "empty_ratio": 0.1})
    assert packet.frame_index == 3
    assert packet.width == 12 and packet.height == 10
    assert packet.triage_metrics["variance"] == 40.0

def test_detection_packet_wraps_corner_detection():
    det = CornerDetection(corners=((0.0, 0.0), (5.0, 0.0), (5.0, 7.0), (0.0, 7.0)), confidence=0.8, metadata={"model": "fake"})
    packet = DetectionPacket(frame_index=9, timestamp_ms=300, width=100, height=200, corner_detection=det)
    assert packet.corner_detection.confidence == 0.8
    assert packet.frame_index == 9

def test_processing_result_has_v21_counts():
    result = ProcessingResult(video_id=1, frame_count=12, accepted_frame_count=7, detection_count=5, saved_instance_count=5, output_dir=Path("out"))
    assert result.saved_instance_count == 5
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/test_models_v21.py -v`  
Expected: FAIL with import/attribute errors for new dataclasses or fields.

- [ ] **Step 3: Implement models and options**

```python
# src/card_capture/models.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

Point = Tuple[float, float]
Polygon = Tuple[Point, Point, Point, Point]

@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray
    width: int
    height: int
    triage_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class CornerDetection:
    corners: Polygon
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DetectionPacket:
    frame_index: int
    timestamp_ms: int
    width: int
    height: int
    corner_detection: CornerDetection

@dataclass(frozen=True)
class ProcessingResult:
    video_id: int
    frame_count: int
    accepted_frame_count: int
    detection_count: int
    saved_instance_count: int
    output_dir: Path
```

```python
# src/card_capture/pipeline.py (options section)
@dataclass(frozen=True)
class ProcessingOptions:
    output_dir: Path
    reader_backend: str = "auto"
    queue_size: int = 64
    inference_batch_size: int = 16
    corner_confidence_threshold: float = 0.5
    blur_threshold: float = 30.0
    variance_threshold: float = 20.0
    empty_pixel_threshold: float = 0.98
```

- [ ] **Step 4: Re-run model tests**

Run: `pytest tests/test_models_v21.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add src/card_capture/models.py src/card_capture/pipeline.py tests/test_models_v21.py
git commit -m "refactor: add v2.1 pipeline packet and result models"
```

### Task 2: Build Ingestion Backends and Triage Filter

**Files:**
- Create: `src/card_capture/ingestion.py`
- Test: `tests/test_ingestion.py`

- [ ] **Step 1: Write failing ingestion tests**

```python
# tests/test_ingestion.py
import numpy as np
from card_capture.ingestion import FrameTriageFilter, _resolve_reader_backend

def test_triage_rejects_empty_frame():
    filt = FrameTriageFilter(blur_threshold=5.0, variance_threshold=5.0, empty_pixel_threshold=0.95)
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)
    accepted, metrics = filt.evaluate(frame)
    assert accepted is False
    assert metrics["empty_ratio"] >= 0.95

def test_triage_accepts_textured_frame():
    filt = FrameTriageFilter(blur_threshold=5.0, variance_threshold=5.0, empty_pixel_threshold=0.99)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, ::2] = 200
    accepted, metrics = filt.evaluate(frame)
    assert accepted is True
    assert metrics["variance"] > 5.0

def test_backend_resolver_auto_falls_back_to_pyav_when_decord_missing(monkeypatch):
    monkeypatch.setattr("card_capture.ingestion._decord_available", lambda: False)
    assert _resolve_reader_backend("auto") == "pyav"
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/test_ingestion.py -v`  
Expected: FAIL because module/functions do not exist.

- [ ] **Step 3: Implement ingestion module**

```python
# src/card_capture/ingestion.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Tuple
import cv2
import numpy as np
from .models import FramePacket

class FrameReader(Protocol):
    def iter_frames(self, video_path: Path) -> Iterator[FramePacket]:
        ...

def _decord_available() -> bool:
    try:
        import decord  # noqa: F401
        return True
    except Exception:
        return False

def _resolve_reader_backend(preferred: str) -> str:
    if preferred not in {"auto", "decord", "pyav"}:
        raise ValueError(f"Unsupported reader backend: {preferred}")
    if preferred == "auto":
        return "decord" if _decord_available() else "pyav"
    return preferred

@dataclass(frozen=True)
class FrameTriageFilter:
    blur_threshold: float
    variance_threshold: float
    empty_pixel_threshold: float

    def evaluate(self, frame: np.ndarray) -> Tuple[bool, dict[str, float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        variance = float(np.var(gray))
        empty_ratio = float(np.mean(gray >= 250))
        accepted = blur >= self.blur_threshold and variance >= self.variance_threshold and empty_ratio < self.empty_pixel_threshold
        return accepted, {"blur": blur, "variance": variance, "empty_ratio": empty_ratio}
```

- [ ] **Step 4: Re-run ingestion tests**

Run: `pytest tests/test_ingestion.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/ingestion.py tests/test_ingestion.py
git commit -m "feat: add v2.1 ingestion backends and triage filter"
```

### Task 3: Replace Detector Layer with ONNX-First Corner Adapters

**Files:**
- Modify: `src/card_capture/detectors.py`
- Modify: `tests/test_detectors.py`

- [ ] **Step 1: Add failing corner-detector tests**

```python
# tests/test_detectors.py
import numpy as np
from card_capture.models import FramePacket
from card_capture.detectors import FakeCornerDetector

def test_fake_corner_detector_returns_one_detection_per_frame():
    frame = FramePacket(frame_index=1, timestamp_ms=40, image=np.zeros((100, 80, 3), dtype=np.uint8), width=80, height=100)
    detector = FakeCornerDetector(confidence=0.9)
    out = detector.detect_batch([frame], confidence_threshold=0.5)
    assert len(out) == 1
    assert out[0].corner_detection.confidence == 0.9

def test_fake_corner_detector_honors_confidence_threshold():
    frame = FramePacket(frame_index=1, timestamp_ms=40, image=np.zeros((100, 80, 3), dtype=np.uint8), width=80, height=100)
    detector = FakeCornerDetector(confidence=0.4)
    assert detector.detect_batch([frame], confidence_threshold=0.5) == []
```

- [ ] **Step 2: Run detector tests and confirm failure**

Run: `pytest tests/test_detectors.py -v`  
Expected: FAIL due to missing `FakeCornerDetector` or `detect_batch`.

- [ ] **Step 3: Implement detector protocol and fake + ONNX adapter**

```python
# src/card_capture/detectors.py
from typing import List, Protocol
from .models import CornerDetection, DetectionPacket, FramePacket

class CornerDetector(Protocol):
    runtime: str
    model_name: str
    def detect_batch(self, frames: List[FramePacket], confidence_threshold: float) -> List[DetectionPacket]:
        ...

class FakeCornerDetector:
    runtime = "fake"
    model_name = "fake-corner-detector"
    def __init__(self, confidence: float = 0.99) -> None:
        self.confidence = confidence
    def detect_batch(self, frames: List[FramePacket], confidence_threshold: float) -> List[DetectionPacket]:
        out: List[DetectionPacket] = []
        for frame in frames:
            if self.confidence < confidence_threshold:
                continue
            corners = ((0.0, 0.0), (float(frame.width - 1), 0.0), (float(frame.width - 1), float(frame.height - 1)), (0.0, float(frame.height - 1)))
            out.append(DetectionPacket(frame.frame_index, frame.timestamp_ms, frame.width, frame.height, CornerDetection(corners=corners, confidence=self.confidence, metadata={"runtime": self.runtime, "model": self.model_name})))
        return out
```

- [ ] **Step 4: Re-run detector tests**

Run: `pytest tests/test_detectors.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/detectors.py tests/test_detectors.py
git commit -m "refactor: replace yolo detector contract with corner detector adapters"
```

### Task 4: Replace Storage with v2.1 Schema and APIs

**Files:**
- Modify: `src/card_capture/storage.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Write failing storage tests against new schema**

```python
# tests/test_storage.py
from pathlib import Path
from card_capture.models import CornerDetection
from card_capture.storage import Storage

def test_storage_v21_records_instance_view_and_evidence(tmp_path: Path):
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    video_id = storage.add_video("/videos/input.mov", "hash", 1000, 1920, 1080)
    instance_id = storage.add_card_instance(video_id, "card_1")
    view_id = storage.add_card_view(instance_id, frame_index=12, timestamp_ms=400, detection=CornerDetection(corners=((0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)), confidence=0.8, metadata={"model": "fake"}))
    evidence_id = storage.add_evidence_frame(view_id, "output/frames/f12.jpg", 1920, 1080, {"blur": 90.0})
    rows = storage.list_card_instances(video_id)
    assert instance_id == 1 and view_id == 1 and evidence_id == 1
    assert rows[0]["track_id"] == "card_1"
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/test_storage.py -v`  
Expected: FAIL due to missing v2.1 APIs/tables.

- [ ] **Step 3: Implement v2.1 tables and storage methods**

```python
# src/card_capture/storage.py (schema excerpt)
CREATE TABLE IF NOT EXISTS card_instances (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  video_id INTEGER NOT NULL REFERENCES videos(id),
  track_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS card_views (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  card_instance_id INTEGER NOT NULL REFERENCES card_instances(id),
  frame_index INTEGER NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  corners_json TEXT NOT NULL,
  confidence REAL NOT NULL,
  rectified_path TEXT,
  quality_score_json TEXT,
  is_canonical INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_frames (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  card_view_id INTEGER NOT NULL REFERENCES card_views(id),
  source_frame_path TEXT NOT NULL,
  frame_width INTEGER NOT NULL,
  frame_height INTEGER NOT NULL,
  metrics_json TEXT NOT NULL
);
```

```python
# src/card_capture/storage.py (API excerpt)
def add_card_instance(self, video_id: int, track_id: str) -> int: ...
def add_card_view(self, card_instance_id: int, frame_index: int, timestamp_ms: int, detection: CornerDetection, rectified_path: str | None = None, quality_score: dict[str, float] | None = None, is_canonical: bool = False) -> int: ...
def add_evidence_frame(self, card_view_id: int, source_frame_path: str, frame_width: int, frame_height: int, metrics: dict[str, float]) -> int: ...
def list_card_instances(self, video_id: int) -> list[dict[str, object]]: ...
```

- [ ] **Step 4: Re-run storage tests**

Run: `pytest tests/test_storage.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/storage.py tests/test_storage.py
git commit -m "refactor: replace legacy tables with v2.1 storage schema"
```

### Task 5: Implement Multiprocessing Producer/Consumer Pipeline

**Files:**
- Modify: `src/card_capture/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add failing multiprocessing pipeline tests**

```python
# tests/test_pipeline.py
from pathlib import Path
from card_capture.pipeline import ProcessingOptions, VideoProcessor
from card_capture.storage import Storage

def test_pipeline_v21_processes_fake_detector_and_persists_instances(tmp_path: Path):
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"fake")
    storage = Storage(tmp_path / "cards.sqlite")
    storage.initialize()
    processor = VideoProcessor(storage=storage)
    result = processor.process(video_path, ProcessingOptions(output_dir=tmp_path / "out", reader_backend="auto", queue_size=8, inference_batch_size=4, corner_confidence_threshold=0.5))
    assert result.video_id == 1
    assert result.detection_count >= 1
    assert result.saved_instance_count >= 1
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/test_pipeline.py -v`  
Expected: FAIL due to constructor/options mismatch.

- [ ] **Step 3: Implement pipeline orchestration**

```python
# src/card_capture/pipeline.py (orchestration excerpt)
ctx = mp.get_context("spawn")
in_q: mp.Queue = ctx.Queue(maxsize=options.queue_size)
out_q: mp.Queue = ctx.Queue(maxsize=options.queue_size)
err_q: mp.Queue = ctx.Queue()
producer = ctx.Process(target=_producer_main, args=(video_path, options, in_q, err_q))
consumer = ctx.Process(target=_consumer_main, args=(options, detector, in_q, out_q, err_q))
producer.start(); consumer.start()
producer.join(); in_q.put(None); consumer.join()
if not err_q.empty():
    raise RuntimeError(err_q.get())
```

```python
# src/card_capture/pipeline.py (persist excerpt)
instance_id = self.storage.add_card_instance(video_id, f"card_{seq}")
view_id = self.storage.add_card_view(instance_id, packet.frame_index, packet.timestamp_ms, packet.corner_detection)
self.storage.add_evidence_frame(view_id, str(source_frame_path), packet.width, packet.height, frame_metrics)
```

- [ ] **Step 4: Re-run pipeline tests**

Run: `pytest tests/test_pipeline.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/pipeline.py tests/test_pipeline.py
git commit -m "feat: add multiprocessing v2.1 producer-consumer pipeline"
```

### Task 6: Replace CLI Process Surface for v2.1

**Files:**
- Modify: `src/card_capture/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI flag/wiring tests**

```python
# tests/test_cli.py
from card_capture.cli import build_parser

def test_process_subparser_accepts_v21_flags():
    parser = build_parser()
    args = parser.parse_args(["process", "video.mov", "--reader-backend", "auto", "--queue-size", "64", "--inference-batch-size", "16", "--corner-confidence", "0.6", "--blur-threshold", "30", "--variance-threshold", "20", "--empty-pixel-threshold", "0.95"])
    assert args.reader_backend == "auto"
    assert args.queue_size == 64
    assert args.inference_batch_size == 16
    assert args.corner_confidence == 0.6
```

- [ ] **Step 2: Run CLI tests and confirm failure**

Run: `pytest tests/test_cli.py -v`  
Expected: FAIL because new flags are missing.

- [ ] **Step 3: Implement CLI v2.1 process options**

```python
# src/card_capture/cli.py (flags excerpt)
process.add_argument("--reader-backend", choices=["auto", "decord", "pyav"], default="auto")
process.add_argument("--queue-size", type=_positive_int, default=64)
process.add_argument("--inference-batch-size", type=_positive_int, default=16)
process.add_argument("--corner-confidence", type=_unit_float, default=0.5)
process.add_argument("--blur-threshold", type=_positive_float, default=30.0)
process.add_argument("--variance-threshold", type=_positive_float, default=20.0)
process.add_argument("--empty-pixel-threshold", type=_unit_float, default=0.98)
process.add_argument("--detector", choices=["docaligner", "fake"], default="docaligner")
```

```python
# src/card_capture/cli.py (_run_process excerpt)
processor = VideoProcessor(storage=storage)
result = processor.process(args.video_path, ProcessingOptions(output_dir=args.output_dir, reader_backend=args.reader_backend, queue_size=args.queue_size, inference_batch_size=args.inference_batch_size, corner_confidence_threshold=args.corner_confidence, blur_threshold=args.blur_threshold, variance_threshold=args.variance_threshold, empty_pixel_threshold=args.empty_pixel_threshold))
```

- [ ] **Step 4: Re-run CLI tests**

Run: `pytest tests/test_cli.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/card_capture/cli.py tests/test_cli.py
git commit -m "refactor: replace process cli surface with v2.1 options"
```

### Task 7: Add Dependencies and Cross-Module Integration Test Pass

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add failing dependency declaration test**

```python
# tests/test_pipeline.py
from pathlib import Path

def test_pyproject_declares_v21_optional_runtime_dependencies():
    text = Path("pyproject.toml").read_text()
    assert "onnxruntime" in text
    assert "decord" in text
    assert "av" in text
```

- [ ] **Step 2: Run targeted tests and confirm failure**

Run: `pytest tests/test_pipeline.py::test_pyproject_declares_v21_optional_runtime_dependencies -v`  
Expected: FAIL because dependencies are missing.

- [ ] **Step 3: Add optional dependency group**

```toml
[project.optional-dependencies]
pipeline_v21 = [
  "decord",
  "av",
  "onnxruntime",
]
```

- [ ] **Step 4: Run focused integration checks**

Run: `pytest tests/test_models_v21.py tests/test_ingestion.py tests/test_detectors.py tests/test_storage.py tests/test_pipeline.py tests/test_cli.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_pipeline.py
git commit -m "build: add v2.1 pipeline runtime dependencies"
```

### Task 8: Update Docs and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `QUICK_REFERENCE.md`

- [ ] **Step 1: Write doc assertions as failing checks**

```python
# tests/test_cli.py
from pathlib import Path

def test_readme_mentions_reader_backend_flag():
    assert "--reader-backend" in Path("README.md").read_text()

def test_quick_reference_mentions_multiprocessing_pipeline():
    assert "producer/consumer" in Path("QUICK_REFERENCE.md").read_text().lower()
```

- [ ] **Step 2: Run checks and confirm failure**

Run: `pytest tests/test_cli.py::test_readme_mentions_reader_backend_flag tests/test_cli.py::test_quick_reference_mentions_multiprocessing_pipeline -v`  
Expected: FAIL before docs update.

- [ ] **Step 3: Update docs for new architecture**

```markdown
# README.md excerpt
card-capture process video.mov \
  --reader-backend auto \
  --queue-size 64 \
  --inference-batch-size 16 \
  --corner-confidence 0.5
```

```markdown
# QUICK_REFERENCE.md excerpt
Video -> Producer (decord/PyAV ingest + triage) -> Queue -> Consumer (ONNX corner detection) -> v2.1 SQLite storage
```

- [ ] **Step 4: Run full test suite**

Run: `pytest -v`  
Expected: PASS.

- [ ] **Step 5: Run smoke command and verify DB content**

Run:
```bash
card-capture process /path/to/video.mov --output-dir card_capture_output_v21 --db card_capture_output_v21/cards.sqlite --reader-backend auto --queue-size 64 --inference-batch-size 16 --corner-confidence 0.5
```
Expected: command exits `0` and writes `videos`, `card_instances`, `card_views`, `evidence_frames` rows.

- [ ] **Step 6: Commit**

```bash
git add README.md QUICK_REFERENCE.md tests/test_cli.py
git commit -m "docs: update usage and architecture notes for v2.1 stage1/2 pipeline"
```

## Plan Self-Review

1. **Spec coverage:**  
   - Decord + PyAV fallback: Task 2  
   - ONNX-first corner detection abstraction: Task 3  
   - Multiprocessing queue architecture: Task 5  
   - Breaking v2.1 schema: Task 4  
   - CLI/runtime wiring and thresholds: Task 6  
   - Verification and docs alignment: Tasks 7-8

2. **Placeholder scan:** No `TBD`, `TODO`, or deferred implementation markers in tasks.

3. **Type consistency:** `FramePacket`, `CornerDetection`, `DetectionPacket`, `ProcessingOptions`, and storage API names are consistent across tasks.
