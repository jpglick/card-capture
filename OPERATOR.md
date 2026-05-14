# Card Capture — Operator Guide

Practical reference for running the pipeline, the web app, and the training
loop. Assumes you are in the repo root with the virtualenv active.

---

## 1. First-time setup

```bash
# Install all dependencies (pipeline + app + models)
pip install -e ".[pipeline_v21,model,app]"

# Confirm the CLI works
card-capture --help
```

The YOLO model downloads automatically on first run from HuggingFace
(`AlecKarfonta/cardcaptor-v3`). If you have `models/presence_classifier.pt`
it will be loaded; if not, the pipeline falls back to an Otsu-based sampler.

---

## 2. Processing videos

### 2.1 The fast path (recommended while you have few videos)

```bash
card-capture process /path/to/video.MOV \
  --output-dir card_capture_output \
  --db card_capture_output/cards.sqlite
```

Everything lands in `card_capture_output/`:
```
card_capture_output/
  frames/           source frames (one per detection)
  crops/            fused canonical images (750×1050 px)
  cards.sqlite      all metadata, events, and embeddings
  run_telemetry.json
  tracker_association_events.json
```

### 2.2 Processing flags worth knowing

```bash
# Slower but more thorough scan
card-capture process video.MOV --fast-scan-fps 30

# Raise/lower the corner detection bar
card-capture process video.MOV --corner-confidence 0.6   # stricter
card-capture process video.MOV --corner-confidence 0.4   # more permissive

# Force session resets more aggressively (rapid card swaps)
card-capture process video.MOV --valley-drop-ratio 0.30

# Foil cards: lower the threshold if foil detection is triggering on non-foils
card-capture process video.MOV \
  --config card_capture_config.json  # put foil_threshold: 80 in there

# Send output somewhere specific
card-capture process video.MOV \
  --output-dir ~/captures/session_01 \
  --db ~/captures/session_01/cards.sqlite
```

### 2.3 Quick sanity check (no YOLO needed)

```bash
# Runs Stage 1 only (sampler) — takes ~35s on a 4K video instead of 2+ min
card-capture sampler sessions /path/to/video.MOV
```

Prints detected presence windows and valley splits. Good for confirming the
sampler is seeing card swaps before committing to a full run.

### 2.4 Config file

Copy `harness/config.example.json` to `card_capture_config.json` and edit:

```json
{
  "corner_confidence": 0.5,
  "background_novelty_threshold": 0.08,
  "centroid_jump_ratio": 0.30,
  "valley_drop_ratio": 0.40,
  "foil_threshold": 50.0,
  "fast_scan_fps": 15.0,
  "min_track_length": 12
}
```

Pass it with `--config card_capture_config.json`. CLI flags override config
file values for one-off experiments.

---

## 3. Running the web app

The app is two processes: a FastAPI backend and a Svelte dev server.

### Terminal 1 — backend

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The app serves the API at `http://localhost:8000/api/v1/` and the Svelte
frontend is proxied through Vite (see Terminal 2). The `--reload` flag
restarts on Python file changes.

To point the app at a specific database:

```bash
DB_PATH=/path/to/cards.sqlite uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Or edit `app/main.py:create_app()` default directly.

### Terminal 2 — frontend

```bash
cd app/web
npm install        # first time only
npm run dev
```

Open `http://localhost:5173` in the browser. Vite proxies API calls to port
8000, so both terminals must be running.

### App sections

| Route | What it's for |
|---|---|
| `/` → Videos | Drag-drop `.MOV` files to register and process them |
| `/runs` | All pipeline runs, status, telemetry |
| `/runs/<id>` | Per-run timeline, cards, rejection log |
| `/cards` | Full card grid with filters |
| `/label` | Truth labeling, Front/Back trainer, dedup clusters |
| `/label/truth` | Per-video ground-truth editing |
| `/label/fb` | Flash-card mode for Front/Back labels |
| `/regression` | Compare a run against a baseline |
| `/settings` | Config preset editor |
| `/training` | Dataset stats, trigger retrains |

### Processing a video through the UI

1. Go to Videos → drag-drop the `.MOV` file (bytes upload, not path).
2. The file is saved to `card_capture_uploads/` automatically.
3. Click **Run** on the video card.
4. Watch progress in `/runs`.

---

## 4. Where to start with training

### 4.1 What models exist

| Model | File | Status |
|---|---|---|
| YOLO corner detector | `models/cardcaptor_v3_best.pt` | Pre-trained, auto-downloaded |
| Presence classifier | `models/presence_classifier.pt` | Optional; Otsu fallback if missing |
| Front/Back classifier | not yet trained | Falls back to longest-track heuristic |

Training only `presence` and `fb_classifier` is in scope right now. YOLO
retraining is out of scope until you have > 500 labeled corner boxes.

### 4.2 With only a few videos: the right order

```
Process videos → Label truth → Label F/B → Export dataset → Train presence
```

**Step 1: process your videos**

```bash
card-capture process video1.MOV --output-dir out --db out/cards.sqlite
card-capture process video2.MOV --output-dir out --db out/cards.sqlite
# (append to the same DB each time)
```

**Step 2: label truth files (for regression, not strictly required for training)**

Open `http://localhost:5173/label/truth`, select a video, mark each detected
card as real/phantom, confirm Front/Back assignment, and mark any missed cards.
This produces a `<video_id>.truth.json` file used by the harness.

You don't need truth files to train — they're for measuring accuracy. Skip
this if you just want to get a presence model going.

**Step 3: label Front/Back examples**

Open `http://localhost:5173/label/fb`. Each card crop appears full size.
Press `F` = Front, `B` = Back, `S` = skip ambiguous. Aim for at least 50
each before the first retrain; 200+ per class before trusting results.

These labels go into the `fb_labels` table in `cards.sqlite`.

**Step 4: export presence training data**

```bash
card-capture dataset export \
  --db out/cards.sqlite \
  --out-dir data/presence_dataset \
  --confidence-floor 0.7   # only high-confidence detections become positives
```

This mines your processed videos for positive frames (card visible) and
negative frames (background/empty stand). Output is an ImageFolder layout:

```
data/presence_dataset/
  positives/   ← frames with a card
  negatives/   ← background frames
```

**Step 5: train the presence classifier**

```bash
card-capture train presence \
  --data data/presence_dataset \
  --out models/presence_classifier.pt \
  --epochs 8 \
  --batch-size 32
```

Takes a few minutes on Apple Silicon. The model is used in Stage 1 (sampler)
on the next run.

**Step 6: trigger F/B retraining via the app**

Go to `http://localhost:5173/training`. The Training page shows how many
F/B labels you have. When you have enough (≥ 50 per class), click
**Retrain** on `fb_classifier`. The job runs in the background; refresh
the page to see completion status.

The F/B classifier weights land in `models/fb_classifier.pt` and take
effect on the next pipeline run.

### 4.3 Checking if training helped

```bash
# Re-process a video you already labeled
card-capture process video1.MOV --output-dir out_v2 --db out_v2/cards.sqlite

# Run the harness against your truth file
card-capture harness run \
  --baseline v1 \
  --db out_v2/cards.sqlite \
  --truth-dir golden_set/
```

You need at least one video labeled as a truth file for this to produce
meaningful numbers. Even one video gives you card recall and side accuracy.

---

## 5. Inspecting results

### Reviewing crops

```bash
open card_capture_output/crops/
```

Each card appears as a fused 750×1050 PNG. Filename encodes the instance ID
and angle.

### Querying the database directly

```bash
sqlite3 card_capture_output/cards.sqlite

-- How many cards per video?
SELECT video_id, angle, COUNT(*) FROM card_instances GROUP BY video_id, angle;

-- Why were candidates rejected?
SELECT event_type, COUNT(*) FROM pipeline_events GROUP BY event_type ORDER BY 2 DESC;

-- Any failed ReID embeddings?
SELECT * FROM pipeline_events WHERE event_type = 'reid_embedding_failed';

-- Cards with no front
SELECT * FROM card_instances WHERE angle IS NULL OR angle = '';
```

### Telemetry

```bash
cat card_capture_output/run_telemetry.json | python3 -m json.tool | grep -E 'stage|count|fps'
cat card_capture_output/tracker_association_events.json | python3 -m json.tool | head -40
```

---

## 6. Tuning for your videos

Start with the **balanced** preset (the default). Only reach for knobs if
you're seeing a specific problem.

| Symptom | Knob | Direction |
|---|---|---|
| Missing cards (low recall) | `corner_confidence` | lower (0.4) |
| Phantom cards (workspace detected as card) | `background_novelty_threshold` | raise (0.10) |
| Rapid card swaps not splitting into separate sessions | `valley_drop_ratio` | lower (0.30) |
| One card tracked as two across a hand movement | `centroid_jump_ratio` | lower (0.20) |
| Foil cards getting median fusion (blurry) | `foil_threshold` | lower (30.0) |
| Non-foil cards getting glare fusion (washed out) | `foil_threshold` | raise (80.0) |
| Too many frames → slow run | `fast_scan_fps` | lower (10.0) |
| Missing fast card flips | `fast_scan_fps` | raise (30.0) |

Save tuned values to `card_capture_config.json` so every run picks them up.
You can also create named presets in the Settings UI and select them per-run.

---

## 7. Running tests

```bash
# Unit tests only (fast, no video needed)
python3 -m pytest tests/ -q \
  --ignore=tests/pipeline/test_path_equivalence.py

# Just the tests that cover your recent changes
python3 -m pytest tests/test_cli.py tests/migrations/ -q
```

Pre-existing failures (not yours) are in:
- `tests/migrations/test_schema.py::test_migrations_are_idempotent`
- `tests/test_wave1_robustness.py` / `test_wave2_robustness.py` (several)
- `tests/pipeline/test_path_equivalence.py` (metaflow dedup step bug)

---

## 8. Common problems

**`assert_migrations_complete` raises at startup**
Run the app once against a fresh DB and the migrations will apply. If it keeps
failing, check `sqlite3 cards.sqlite "SELECT * FROM _migrations"` to see
which file wasn't applied and why.

**YOLO model not found**
The model downloads automatically if `huggingface-hub` is installed
(`pip install -e ".[model]"`). If you're offline, copy
`cardcaptor_v3_best.pt` into `models/`.

**MPS unavailable warning**
The pipeline asks if you want to continue on CPU. Type `y`. Add
`"device": "cpu"` to `card_capture_config.json` to skip the prompt.

**Crops look rotated 180°**
Your camera is mounted upside-down. Add `"rotate_180": true` to
`card_capture_config.json`.

**All cards labelled "Front" (no Backs detected)**
The F/B classifier isn't trained yet. The pipeline falls back to
"longest-track = Front." Train the F/B classifier (section 4.2) to fix.

**Drag-drop in the UI does nothing / 500 error**
The `card_capture_uploads/` directory is created automatically on first
request. If you see a 500, check the uvicorn terminal for the error — it's
usually a permission issue or a stale DB migration.
