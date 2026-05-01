# Sports Card Video Capture Design

Date: 2026-05-01

## Goal

Build a local-first application that takes a local video file, detects a single sports/trading card being held or moved in the video, and saves the best still images of that card. The first version stores image files on disk and searchable metadata in SQLite. A simple local review UI lets the user accept or reject saved results.

## Scope

Version 1 focuses on local video files, not live camera capture. It prioritizes one visible card at a time. The app detects and extracts card images, but it does not identify the card, read text, grade condition, estimate value, or authenticate the card.

## Recommended Approach

Use a pretrained trading-card detector as the primary detection method:

- Model: `AlecKarfonta/cardcaptor-v3`
- Source: Hugging Face
- Task: single-class `trading_card` object detection
- Output: oriented bounding boxes, useful for tilted cards
- Artifacts: PyTorch checkpoint and ONNX export
- Listed license: MIT

The first implementation can use the Ultralytics/PyTorch path if it loads the model cleanly. The detection interface must remain runtime-neutral so ONNX Runtime can replace Ultralytics later without changing the video pipeline, storage schema, or review UI.

Ultralytics runtime licensing needs review before commercial or proprietary distribution. For a local prototype, ease of integration is more important than final distribution posture.

## Architecture

### CLI

The command-line interface accepts a local video path and processing options such as sample rate, output directory, confidence threshold, and maximum saved candidates.

Example shape:

```text
card-capture process ./input/video.mov --sample-fps 5 --max-candidates 10
card-capture review
```

### VideoSampler

`VideoSampler` reads the source video with OpenCV and emits sampled frames. Each emitted frame includes:

- source video ID
- frame index
- timestamp in milliseconds
- image array
- original frame dimensions

The sample rate is configurable. A default between 3 and 10 FPS is appropriate for v1 because the subject is a card held in view for multiple frames.

### CardDetector

`CardDetector` runs the pretrained model against sampled frames and returns normalized detections:

- polygon corners or oriented bounding box
- confidence score
- class label
- model/runtime metadata

The interface hides whether inference uses Ultralytics/PyTorch or ONNX Runtime. This prevents model runtime choices from leaking into crop, scoring, storage, or UI code.

### CardCropper

`CardCropper` converts each oriented detection into a deskewed card crop. It applies perspective correction based on the detected corners and writes a clean rectangular image. It may also save the source full frame for review and debugging.

Crop output includes:

- crop image path
- optional source frame image path
- card polygon in source-frame coordinates
- crop dimensions
- transform metadata

### QualityScorer

`QualityScorer` assigns a readable-detail score to each crop. The score is a weighted combination of measurable image signals:

- sharpness / blur, using Laplacian variance or a comparable focus metric
- card resolution and frame coverage
- detector confidence
- perspective/crop quality
- glare or overexposure penalty
- duplicate similarity penalty for nearby frames

The scorer stores both the final score and component scores so the review UI can explain why an image was selected.

### CandidateSelector

`CandidateSelector` groups detections that likely represent the same held-card moment. In v1, grouping can be based on timestamp proximity and visual similarity. It then keeps the highest-scoring candidates from each group, subject to a per-video maximum.

This avoids saving many near-identical frames from a single pass through the video.

### Storage

Images are stored as files. Metadata is stored in SQLite.

Suggested tables:

- `videos`: source path, file hash, duration, dimensions, processing status
- `detections`: video ID, timestamp, frame index, detector confidence, polygon, score components, crop path, source frame path
- `saved_cards`: selected detection ID, final image path, final score, accepted/rejected state
- `review_decisions`: saved card ID, decision, notes, created timestamp

The SQLite database is the index. The image folder remains inspectable and portable.

### Review UI

The review UI is a local web app started from the CLI. A lightweight FastAPI app with server-rendered templates is enough for v1.

The UI shows:

- selected card crops
- score and score breakdown
- source timestamp and source frame
- accept/reject controls
- basic filtering by video and review status

The UI should not be responsible for running heavy video processing. Processing happens through the CLI; the UI reads from SQLite and updates review decisions.

## Data Flow

1. User runs the process command with a local video file.
2. The app records the video in SQLite.
3. `VideoSampler` emits frames at the configured sample rate.
4. `CardDetector` detects card-oriented bounding boxes.
5. `CardCropper` deskews and saves card crops.
6. `QualityScorer` calculates final and component scores.
7. `CandidateSelector` chooses the best candidates.
8. `Storage` writes image files and metadata.
9. User starts the review UI and accepts or rejects saved results.

## Error Handling

The app should fail clearly when:

- the video file does not exist or cannot be decoded
- the model artifact cannot be downloaded or loaded
- no card is detected above the confidence threshold
- the output directory is not writable
- SQLite cannot be opened or migrated

Per-frame failures should be logged and skipped when possible. A single bad frame should not fail the whole video.

## Testing Strategy

The first implementation should include focused tests around components that do not require the full model:

- SQLite schema creation and metadata writes
- candidate grouping and selection
- quality score composition
- crop geometry helpers with synthetic polygons
- CLI argument parsing and error handling

Model inference should be wrapped behind an interface so it can be tested with fake detections. A small fixture video or fixture frame set can be added later for end-to-end smoke testing.

## Open Risks

- Real video lighting, glare, sleeves, top loaders, and hand occlusion may reduce detection and crop quality.
- The model is a trading-card detector, not a sports-card identifier.
- Ultralytics runtime licensing should be reviewed before any commercial distribution.
- Quality scoring will need empirical tuning against real user videos.

## Out Of Scope For Version 1

- live webcam capture
- multiple cards visible at once
- card identification or OCR
- price lookup
- condition grading
- cloud sync
- mobile app packaging
