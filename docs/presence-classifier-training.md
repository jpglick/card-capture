# Presence Classifier Training Guide

The presence classifier (`models/presence_classifier.pt`) is a MobileNetV3-Small binary
classifier that predicts P(card present) for each 192px scan frame.  The sampler uses
it to skip empty-workspace frames before running expensive YOLO detection.

---

## Why retraining is needed

The current model was trained exclusively on videos where cards are **set down on a
stand**.  It has never seen hand-held presentation footage and confidently predicts
"no card" for every frame in those videos.  Once retrained with hand-held examples it
will handle both styles correctly.

---

## Data requirements

### Volume
| Class | Minimum | Target |
|---|---|---|
| Positive (card present) | 500 frames | 1 500–2 000 frames |
| Negative (no card) | 500 frames | 1 500–2 000 frames |

Aim for rough class balance.  More data always helps, but quality matters more than
quantity — avoid near-duplicate frames from the same moment in a video.

### Positive class — what to include

| Scenario | Priority | Notes |
|---|---|---|
| Hand holding card toward camera | **Critical** | Single card, finger may partially cover corner |
| Hand holding card at various angles | High | Tilted ±30°, rotated, slightly blurred |
| Card laid flat on stand/workspace | High | Already in training data; keep these |
| Card in sleeve or top-loader | Medium | Common in collection videos |
| Partially occluded card | Medium | Hand covering up to 25% of surface |
| Two cards fanned in hand | Low | Edge case but realistic |

### Negative class — what to include

| Scenario | Priority | Notes |
|---|---|---|
| Empty stand / workspace | **Critical** | Already in training data; keep these |
| Empty hand (no card) | High | Between card presentations |
| Background only (table, shelf) | High | Pan shots, repositioning moments |
| Card completely obscured by hand | Medium | Looks like fist / closed hand |
| Transition blur frames | Low | Hand moving fast between cards |

### What to avoid
- Frames where it is genuinely ambiguous whether a card is present
- Near-duplicates from the same still hold (sample ≤ 1 frame per 0.5 s of stable hold)
- Heavily overexposed or underexposed frames (these teach the wrong texture features)

---

## Step-by-step instructions

### 1. Record / collect source videos

You need videos in both styles:
- **Stand style**: card placed on workspace, held still, picked up (already have IMG_5872-style)
- **Hand-held style**: card presented directly to camera, finger(s) may rest on surface

Aim for **10–20 videos per style**, each 10–60 seconds.  Variety in:
- Card design (sports, Pokémon, foil, matte)
- Background (desk, table, shelf, dark, light)
- Lighting (daylight, indoor, mixed)
- Hand position and grip

### 2. Export frames from the existing pipeline

```bash
# Export presence training data from already-processed videos
card-capture dataset export --db card_capture_output/cards.sqlite --out-dir data/presence_dataset

# Or export from a specific run
card-capture dataset export --db card_capture_output/cards.sqlite --run-id <run_id> --out-dir data/presence_dataset
```

This populates `data/presence_dataset/positives/` and `data/presence_dataset/negatives/`
with 192×(auto-height) JPEG thumbnails already labelled from pipeline metadata.

### 3. Add hand-held frames manually

For hand-held videos that the pipeline currently misses, extract frames directly:

```bash
# Extract one frame per second from a hand-held video
python3 - <<'EOF'
import cv2, pathlib, sys

video = pathlib.Path(sys.argv[1])
out   = pathlib.Path("data/presence_dataset/positives")
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
stride = max(1, int(fps))          # one frame per second
idx = 0
saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % stride == 0:
        small = cv2.resize(frame, (192, int(frame.shape[0] * 192 / frame.shape[1])))
        cv2.imwrite(str(out / f"{video.stem}_{idx:06d}.jpg"), small)
        saved += 1
    idx += 1
cap.release()
print(f"Saved {saved} frames to {out}")
EOF path/to/hand_held_video.MOV
```

Put card-present frames in `positives/` and empty / no-card frames in `negatives/`.
You can also just copy frames directly if you know which ones are which.

### 4. Review and prune the dataset

Spot-check both folders.  Remove:
- Mislabelled frames (card in negatives, no-card in positives)
- Near-duplicates (back-to-back frames of an identical still)
- Extreme blur or near-black frames

A quick way to scan:
```bash
# Open positives in a Quick Look grid (macOS)
open data/presence_dataset/positives/
```

### 5. Train the classifier

```bash
card-capture train presence \
    --data  data/presence_dataset \
    --out   models/presence_classifier.pt \
    --epochs 30 \
    --batch  64
```

Training takes 5–15 minutes on Apple Silicon (MPS).  The script prints per-epoch
validation accuracy.  Stop early if val accuracy plateaus above 95 %.

Typical healthy numbers after retraining with hand-held data:
- Validation accuracy: 96–99 %
- Loss converges by epoch 15–20

### 6. Validate on held-out videos

Run the pipeline on one stand-style and one hand-held video before committing the model:

```bash
# Hand-held video — should now find cards
card-capture process path/to/hand_held.MOV \
    --output-dir /tmp/test_handheld --db /tmp/test_handheld/cards.sqlite

# Stand video — should still work
card-capture process path/to/stand_style.MOV \
    --output-dir /tmp/test_stand --db /tmp/test_stand/cards.sqlite
```

Check `run_telemetry.json` in each output directory:
- `last_fallback_used` should be `false` for both (windows found by classifier)
- `last_selected_frame_count` should be in the hundreds, not single digits
- `saved_instances` should match what you expect

### 7. Adjust presence_threshold if needed

The default threshold is `0.4` (stored in `card_capture_config.json` as
`presence_threshold`).  After retraining you may want to raise it slightly to
re-enable filtering of empty-workspace frames:

```json
{ "presence_threshold": 0.3 }
```

Lower = more frames pass (safer for hand-held), Higher = stricter filtering (faster
for stand-style).  During development, `0.0` is set in config so the classifier
score is not used as a gate — all frames pass and YOLO does the filtering.  Once
the classifier is retrained and validated you can raise this back to `0.3`–`0.4`.

---

## What the classifier sees

Input: 192×(auto) BGR frame → resized/cropped to 224×224 → normalized with ImageNet
mean/std → MobileNetV3-Small → softmax → class 1 = P(card present).

The model runs on every scan frame (15 fps down-sampled scan, not full 60 fps
source), so it processes ~150–500 frames per minute of video.

---

## Files

| Path | Purpose |
|---|---|
| `models/presence_classifier.pt` | Trained weights (state_dict + metadata) |
| `data/presence_dataset/positives/` | Card-present training images (192px) |
| `data/presence_dataset/negatives/` | No-card training images (192px) |
| `src/card_capture/presence/classifier.py` | Inference wrapper |
| `src/card_capture/cli.py` | `train presence` command entry point |
