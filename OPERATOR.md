# Card Capture — Operator Guide

Practical reference for running the pipeline, the web app, and the training
loop. Assumes you are in the repo root with the virtualenv active.

---

## 1. First-time setup

```bash
# Install all dependencies (pipeline + app + models)
pip install -e ".[legacy_tracking,model,app]"

# Confirm the CLI works
card-capture --help
```

The YOLO model downloads automatically on first run from HuggingFace
(`AlecKarfonta/cardcaptor-v3`). If you have `models/presence_classifier.pt`
it will be loaded; if not, the pipeline falls back to a StrideSampler.

---

## 2. Processing videos

### 2.1 The fast path (recommended while you have few videos)

```bash
card-capture process /path/to/video.MOV \
  --output-dir out \
  --db out/cards.sqlite
```

Everything lands in `out/`:
```
out/
  crops/            fused canonical images (750×1050 px)
  cards.sqlite      all metadata, events, and embeddings
  run_telemetry.json
```

### 2.2 Processing flags worth knowing

```bash
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
# Runs Stage 1 only (sampler)
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
  "min_track_length": 3
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
| `/cards` | Full card grid with filters |
| `/label` | Truth labeling, Front/Back trainer, dedup clusters |
| `/settings` | Config preset editor |
| `/training` | Dataset stats, trigger retrains |

---

## 4. Where to start with training

### 4.1 What models exist

| Model | File | Status |
|---|---|---|
| YOLO corner detector | `models/cardcaptor_v3_best.pt` | Pre-trained, auto-downloaded |
| Presence classifier | `models/presence_classifier.pt` | Optional; StrideSampler fallback if missing |
| Front/Back classifier | `models/presence_classifier.pt` | Used for side prediction |

### 4.2 Training workflow

1.  **Process videos:** Append to a single database.
2.  **Label Front/Back:** Use the UI to label examples.
3.  **Export dataset:** Mine processed videos for training data.
4.  **Train:** Train the presence or F/B classifier.

---

## 5. Inspecting results

### Reviewing crops

```bash
open out/crops/
```

### Querying the database directly

```bash
sqlite3 out/cards.sqlite

-- How many cards per video?
SELECT video_id, angle, COUNT(*) FROM card_instances GROUP BY video_id, angle;

-- Why were candidates rejected?
SELECT event_type, COUNT(*) FROM pipeline_events GROUP BY event_type ORDER BY 2 DESC;
```

---

## 6. Tuning for your videos

| Symptom | Knob | Direction |
|---|---|---|
| Missing cards (low recall) | `corner_confidence` | lower (0.4) |
| Phantom cards (workspace detected as card) | `background_novelty_threshold` | raise (0.10) |
| Rapid card swaps not splitting | `valley_drop_ratio` | lower (0.30) |
| Too many frames → slow run | `fast_scan_fps` | lower (10.0) |

---

## 7. Running tests

```bash
# Unit tests only (fast, no video needed)
python3 -m pytest tests/ -q -m "not quarantine"
```

---

## 8. Common problems

**MPS unavailable warning**
The pipeline asks if you want to continue on CPU. Type `y`. Add
`"device": "cpu"` to `card_capture_config.json` to skip the prompt. CUDA is not supported.

**Crops look rotated 180°**
Your camera is mounted upside-down. Add `"rotate_180": true` to
`card_capture_config.json`.
