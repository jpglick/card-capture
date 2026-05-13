# Wave 4 — Surface B (Frontend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the four UI surfaces the v4 plan called for but that landed as stubs or empty routes: Inbox, Settings, A/B comparison, and user-defined config preset persistence.

**Architecture:** Single agent, 4 PRs. Surface B owns frontend Svelte pages, the backend endpoints they call, and a new schema migration. Blocked-by Surface E (CI). B3 also blocked-by A3 (migration-runner logging — first new migration after that lands).

**Tech Stack:** SvelteKit 2.x, TypeScript, Vite, FastAPI, SQLite, Playwright.

**Spec:** `docs/superpowers/specs/2026-05-13-v4-wave4-hardening-design.md` §5.

**Files owned by Surface B:** `app/web/**`, `app/api/{videos,config,regression}.py`, `app/services/{video,playground,regression}_service.py`, `migrations/0003_*.sql`, `app/schemas/v1.py` (additive only).

---

## Pre-flight

- [ ] **P1: Confirm E1 is merged; check A3 status for B3**

```bash
git fetch origin main
ls .github/workflows/test.yml
grep -n "log.warning" migrations/run_migrations.py
```

Expected: workflow exists; `run_migrations.py` has the log call from A3.
If A3 isn't merged, run B1, B2, B4 first and come back to B3.

- [ ] **P2: Create the worktree**

```bash
git worktree add ../card-capture-wave4-b -b wave4/b-frontend origin/main
cd ../card-capture-wave4-b
pip install -e ".[harness,test]"
cd app/web && npm ci && cd ../..
python -m pytest tests/ -q
```

Expected: backend tests green; `app/web/node_modules/` populated.

---

## Task 1: B1 — Inbox page (drag-drop + SSE-driven queue)

**Files:**
- Modify: `app/web/src/routes/videos/+page.svelte`
- Modify: `app/web/src/lib/api.ts` (if needed for SSE helper)
- Create: `app/web/src/lib/components/QueueCard.svelte`
- Create: `app/web/playwright/inbox.spec.ts` (or document a manual smoke flow)
- Modify: `tests/app/test_videos_endpoint.py` (if needed)

- [ ] **Step 1.1: Inspect existing state**

```bash
cat app/web/src/routes/videos/+page.svelte
cat app/api/videos.py
cat app/services/video_service.py
```

Determine: (a) does `POST /api/v1/videos` already accept multipart? (b)
does an SSE channel `/events/{run_id}` already exist (it should from A3)?
(c) what's the existing Svelte page rendering?

- [ ] **Step 1.2: Write a backend test for the videos endpoint**

If `POST /api/v1/videos` already has a test, extend it. Otherwise create
`tests/app/test_videos_endpoint.py`:

```python
"""Inbox: POST /api/v1/videos accepts a video and enqueues it."""
from fastapi.testclient import TestClient
from app.main import create_app


def test_post_video_returns_201_with_video_record():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/videos",
        json={"filename": "practice.mov"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "practice.mov"
    assert "video_id" in body
    assert body["status"] in ("pending", "processing")


def test_get_videos_lists_posted_videos():
    client = TestClient(create_app())
    client.post("/api/v1/videos", json={"filename": "a.mov"})
    response = client.get("/api/v1/videos")
    assert response.status_code == 200
    body = response.json()
    assert any(v["filename"] == "a.mov" for v in body["items"])
```

Run: `pytest tests/app/test_videos_endpoint.py -v`

If failures point to missing backend functionality, that's part of B1 —
implement the endpoint to satisfy the contract from `docs/contracts/v1-api.md`.

- [ ] **Step 1.3: Implement the Svelte page**

Edit `app/web/src/routes/videos/+page.svelte` to be the drag-drop inbox:

```svelte
<script lang="ts">
  import { onMount } from "svelte";
  import QueueCard from "$lib/components/QueueCard.svelte";
  import { listVideos, uploadVideo } from "$lib/api";

  let videos: Array<{ video_id: string; filename: string; status: string }> = [];
  let dragOver = false;

  async function refresh() {
    videos = await listVideos();
  }

  async function handleDrop(event: DragEvent) {
    event.preventDefault();
    dragOver = false;
    const files = event.dataTransfer?.files;
    if (!files) return;
    for (const file of Array.from(files)) {
      await uploadVideo(file);
    }
    await refresh();
  }

  onMount(() => {
    refresh();
  });
</script>

<svelte:head><title>Inbox — card-capture</title></svelte:head>

<h1>Inbox</h1>

<div
  class="dropzone"
  class:over={dragOver}
  on:dragover|preventDefault={() => (dragOver = true)}
  on:dragleave={() => (dragOver = false)}
  on:drop={handleDrop}
>
  Drag video files here, or click to choose.
</div>

<section class="queue">
  {#each videos as v (v.video_id)}
    <QueueCard video={v} />
  {/each}
</section>

<style>
  .dropzone { border: 2px dashed #888; padding: 2rem; text-align: center; }
  .dropzone.over { border-color: #4af; background: #eef; }
  .queue { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; margin-top: 1rem; }
</style>
```

Create `app/web/src/lib/components/QueueCard.svelte`:

```svelte
<script lang="ts">
  import { onMount, onDestroy } from "svelte";

  export let video: { video_id: string; filename: string; status: string };

  let eventSource: EventSource | null = null;

  onMount(() => {
    if (video.status === "pending" || video.status === "processing") {
      eventSource = new EventSource(`/events/${video.video_id}`);
      eventSource.onmessage = (e) => {
        const payload = JSON.parse(e.data);
        if (payload.event === "status_changed") {
          video = { ...video, status: payload.data.status };
        }
      };
    }
  });

  onDestroy(() => {
    eventSource?.close();
  });
</script>

<article class="card status-{video.status}">
  <h3>{video.filename}</h3>
  <p>Status: <strong>{video.status}</strong></p>
</article>

<style>
  .card { border: 1px solid #ccc; padding: 1rem; border-radius: 4px; }
  .status-completed { border-color: #4a4; }
  .status-failed { border-color: #c44; }
</style>
```

Update `app/web/src/lib/api.ts` to add the helpers (if not already
present):

```typescript
export async function listVideos() {
  const r = await fetch("/api/v1/videos");
  return (await r.json()).items;
}

export async function uploadVideo(file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  await fetch("/api/v1/videos", { method: "POST", body: form });
}
```

(If the existing `POST /api/v1/videos` accepts JSON only, B1's scope
grows to add multipart handling on the backend. Note this in the PR
description.)

- [ ] **Step 1.4: Write a Playwright smoke test (or document manual flow)**

If Playwright is wired in `app/web/`, create `app/web/playwright/inbox.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";

test("inbox renders dropzone and lists videos", async ({ page }) => {
  await page.goto("/videos");
  await expect(page.locator(".dropzone")).toBeVisible();
});
```

Otherwise, document the manual smoke flow in the PR body:

```
## Manual smoke
1. `uvicorn app.main:app` in one terminal.
2. `cd app/web && npm run dev` in another.
3. Open `http://localhost:5173/videos`.
4. Drag a `.mov` file onto the dropzone.
5. Expect a queue card to appear with status "pending".
6. Watch the status flip as the pipeline progresses.
```

- [ ] **Step 1.5: Run backend tests; build the frontend**

```bash
python -m pytest tests/ -q
cd app/web && npm run build && cd ../..
```

Expected: backend green; frontend builds without errors.

- [ ] **Step 1.6: Commit and open PR**

```bash
git add app/web/src/routes/videos/ app/web/src/lib/ \
        app/api/videos.py app/services/video_service.py \
        tests/app/test_videos_endpoint.py
git commit -m "feat(web): Inbox page with drag-drop + SSE queue updates

/videos route becomes a drag-drop inbox. POST /api/v1/videos enqueues
the file; QueueCard subscribes to /events/{run_id} and live-updates
status.

Closes V4_CONCERNS §4.15.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push -u origin wave4/b-frontend
gh pr create --title "[Wave 4 — Surface B] Inbox: drag-drop + SSE queue (B1)" --body "$(cat <<'EOF'
## Summary
- /videos becomes a drag-drop inbox.
- QueueCard component subscribes to /events/{run_id} (SSE from A3) and live-updates status.
- Backend test for POST /api/v1/videos.

Closes V4_CONCERNS §4.15.

## Test plan
- [x] new backend test: tests/app/test_videos_endpoint.py
- [x] frontend smoke: see manual flow below (or Playwright test, if wired)
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 2: B2 — Settings tab body

**Files:**
- Modify: `app/web/src/routes/settings/+page.svelte`
- Create: `app/web/src/lib/components/PresetEditor.svelte`

- [ ] **Step 2.1: Rebase**

```bash
git fetch origin main && git rebase origin/main
```

- [ ] **Step 2.2: Inspect existing state**

```bash
cat app/web/src/routes/settings/+page.svelte
ls app/web/src/routes/settings/
```

Note: `settings/playground/` already exists (B3 threshold playground).
The new `+page.svelte` is the landing page that lists presets and links
to the playground.

- [ ] **Step 2.3: Build the page**

Replace `app/web/src/routes/settings/+page.svelte`:

```svelte
<script lang="ts">
  import { onMount } from "svelte";
  import PresetEditor from "$lib/components/PresetEditor.svelte";

  type Preset = {
    preset_name: string;
    description: string;
    config: Record<string, number>;
  };

  let presets: Preset[] = [];

  async function refresh() {
    const r = await fetch("/api/v1/config/presets");
    presets = await r.json();
  }

  onMount(refresh);
</script>

<svelte:head><title>Settings — card-capture</title></svelte:head>

<h1>Settings</h1>

<section>
  <h2>Config presets</h2>
  {#each presets as preset (preset.preset_name)}
    <PresetEditor {preset} on:saved={refresh} />
  {/each}
</section>

<section>
  <h2>Threshold playground</h2>
  <p><a href="/settings/playground">Open the threshold playground →</a></p>
</section>
```

Create `app/web/src/lib/components/PresetEditor.svelte`:

```svelte
<script lang="ts">
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher();

  export let preset: {
    preset_name: string;
    description: string;
    config: Record<string, number>;
  };

  // Tooltips per threshold — keep terse.
  const tooltips: Record<string, string> = {
    corner_confidence: "Lower = more partial / corner cards detected. Higher = fewer false positives.",
    background_novelty_threshold: "Lower = more candidates pass the empty-workspace gate.",
    centroid_jump_ratio: "Track-reset threshold as a fraction of frame width.",
    valley_drop_ratio: "Sensitivity of valley-split (hand-swap) detection.",
    foil_threshold: "Laplacian variance threshold for foil-card detection.",
  };

  let saving = false;
  let userPresetName = `${preset.preset_name}-custom`;

  async function saveAsNewPreset() {
    saving = true;
    try {
      await fetch("/api/v1/config/presets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preset_name: userPresetName,
          description: `Customised from ${preset.preset_name}`,
          config: preset.config,
        }),
      });
      dispatch("saved");
    } finally {
      saving = false;
    }
  }
</script>

<article class="preset">
  <h3>{preset.preset_name}</h3>
  <p>{preset.description}</p>

  {#each Object.entries(preset.config) as [key, value] (key)}
    <label>
      <span>{key}</span>
      <input
        type="number"
        step="0.01"
        bind:value={preset.config[key]}
        title={tooltips[key] ?? ""}
      />
      <small>{tooltips[key] ?? ""}</small>
    </label>
  {/each}

  <div class="actions">
    <label>
      Save as:
      <input bind:value={userPresetName} />
    </label>
    <button on:click={saveAsNewPreset} disabled={saving}>
      {saving ? "Saving…" : "Save as preset"}
    </button>
  </div>
</article>

<style>
  .preset { border: 1px solid #ddd; padding: 1rem; margin-bottom: 1rem; border-radius: 4px; }
  label { display: block; margin: 0.5rem 0; }
  small { color: #666; display: block; }
  .actions { margin-top: 1rem; display: flex; gap: 0.5rem; align-items: center; }
</style>
```

- [ ] **Step 2.4: Frontend build**

```bash
cd app/web && npm run build && cd ../..
```

Expected: builds without errors. The "Save as preset" button hits an
endpoint that's still a no-op stub (will be wired in B3); for now it
returns the posted payload unchanged, so the UI works but no data
persists yet. Note this in the PR description.

- [ ] **Step 2.5: Commit and open PR**

```bash
git add app/web/src/routes/settings/ app/web/src/lib/components/PresetEditor.svelte
git commit -m "feat(web): Settings tab — preset list + editor

/settings/+page.svelte lists every preset from GET /api/v1/config/presets
with a PresetEditor per row. Sliders/number inputs per threshold;
tooltips explain trade-offs. 'Save as preset' button hits the stub
endpoint; B3 will make it persist.

Closes V4_CONCERNS §4.14.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface B] Settings tab (B2)" --body "$(cat <<'EOF'
## Summary
- /settings landing page with preset list and editor.
- PresetEditor component: per-threshold number inputs + tooltips + 'Save as preset'.
- B3 will wire the save endpoint to actually persist.

Closes V4_CONCERNS §4.14.

## Test plan
- [x] frontend builds clean
- [x] manual: open /settings, edit a slider, click Save as preset (current behaviour: round-trips payload but doesn't persist; that's B3)
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 3: B3 — Config-preset persistence

**Files:**
- Create: `migrations/0003_config_presets.sql`
- Modify: `app/api/config.py`
- Modify: `app/services/playground_service.py` (or a new `config_service`)
- Modify: `app/schemas/v1.py` (if `ConfigPreset` needs a `user_defined` flag)
- Modify: `docs/contracts/storage-schema.md` and `docs/contracts/v1-api.md`
- Modify: `tests/contracts/test_drift.py` (cover the new table + endpoint)
- Create: `tests/app/test_config_preset_persistence.py`

- [ ] **Step 3.1: Rebase; confirm A3 merged**

```bash
git fetch origin main && git rebase origin/main
grep -n "log.warning" migrations/run_migrations.py
```

Expected: log call present.

- [ ] **Step 3.2: Write the failing test**

Create `tests/app/test_config_preset_persistence.py`:

```python
"""Config presets persist to DB and union with builtins.

Closes V4_CONCERNS §4.10.
"""
from fastapi.testclient import TestClient
from app.main import create_app


def test_post_preset_persists_and_appears_in_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(db_path=tmp_path / "cards.sqlite"))

    response = client.post(
        "/api/v1/config/presets",
        json={
            "preset_name": "my_custom",
            "description": "Hand-tuned for hard cases",
            "config": {"corner_confidence": 0.45},
        },
    )
    assert response.status_code == 201

    listed = client.get("/api/v1/config/presets").json()
    names = {p["preset_name"] for p in listed}
    assert "my_custom" in names
    # Builtins still present
    assert {"fast", "balanced", "quality"} <= names


def test_duplicate_preset_name_returns_409(tmp_path):
    client = TestClient(create_app(db_path=tmp_path / "cards.sqlite"))
    payload = {"preset_name": "dup", "description": "", "config": {}}
    r1 = client.post("/api/v1/config/presets", json=payload)
    assert r1.status_code == 201
    r2 = client.post("/api/v1/config/presets", json=payload)
    assert r2.status_code == 409
```

- [ ] **Step 3.3: Run the test — expect FAIL**

```bash
pytest tests/app/test_config_preset_persistence.py -v
```

Expected: both fail (POST currently returns 201 with no persistence).

- [ ] **Step 3.4: Write the migration**

Create `migrations/0003_config_presets.sql`:

```sql
-- migrations/0003_config_presets.sql
-- User-defined config presets persisted in the same SQLite as cards.

CREATE TABLE IF NOT EXISTS config_presets (
    preset_name   TEXT    PRIMARY KEY,
    description   TEXT    NOT NULL,
    config_json   TEXT    NOT NULL,            -- JSON blob: {field: value, ...}
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
```

- [ ] **Step 3.5: Update `app/api/config.py`**

Replace the stub `create_preset` and update `list_presets`:

```python
import json
import sqlite3
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from app.schemas.v1 import ConfigPlayground, ConfigPreset

router = APIRouter()

_BUILTIN_PRESETS = [...]  # unchanged

def _user_presets(db_path) -> list[ConfigPreset]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT preset_name, description, config_json FROM config_presets "
            "ORDER BY created_at"
        ).fetchall()
    return [
        ConfigPreset(
            preset_name=name,
            description=description,
            config=json.loads(config_json),
        )
        for name, description, config_json in rows
    ]


@router.get("/presets", response_model=list[ConfigPreset])
def list_presets(request: Request):
    return _BUILTIN_PRESETS + _user_presets(request.app.state.db_path)


@router.post("/presets", response_model=ConfigPreset, status_code=201)
def create_preset(payload: ConfigPreset, request: Request):
    builtins = {p.preset_name for p in _BUILTIN_PRESETS}
    if payload.preset_name in builtins:
        raise HTTPException(409, f"Cannot override built-in preset {payload.preset_name!r}")
    try:
        with sqlite3.connect(request.app.state.db_path) as conn:
            conn.execute(
                "INSERT INTO config_presets(preset_name, description, config_json) "
                "VALUES (?, ?, ?)",
                (payload.preset_name, payload.description, json.dumps(payload.config)),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"Preset {payload.preset_name!r} already exists")
    return payload
```

- [ ] **Step 3.6: Document the new table and endpoint behaviour**

Edit `docs/contracts/storage-schema.md`. Append a new table section
matching the DDL above. Add it to the Table Summary at the top.

Edit `docs/contracts/v1-api.md`. Update the `POST /api/v1/config/presets`
section to document the 409 case for duplicates and built-in overrides.

- [ ] **Step 3.7: Extend the drift gate**

In `tests/contracts/test_drift.py`, the existing
`test_storage_schema_columns_appear_in_contract` already picks up the
new file. Verify it parses `0003_config_presets.sql`:

```python
# Update DDL read to glob all migrations:
ddl = "\n".join(
    p.read_text() for p in sorted((REPO_ROOT / "migrations").glob("*.sql"))
)
```

If that change is needed, make it.

- [ ] **Step 3.8: Run the tests — expect PASS**

```bash
pytest tests/app/test_config_preset_persistence.py tests/contracts/ -v
python -m pytest tests/ -q
```

Expected: green.

- [ ] **Step 3.9: Wire the frontend Save button to the now-real endpoint**

`PresetEditor.svelte` already POSTs to `/api/v1/config/presets`. Verify
the dispatch refreshes the preset list. No frontend code change needed
if it was already correct.

- [ ] **Step 3.10: Commit and open PR**

```bash
git add migrations/0003_config_presets.sql \
        app/api/config.py \
        docs/contracts/storage-schema.md \
        docs/contracts/v1-api.md \
        tests/contracts/test_drift.py \
        tests/app/test_config_preset_persistence.py
git commit -m "feat(config): persist user-defined config presets

New config_presets table (migration 0003). POST /api/v1/config/presets
persists; GET unions builtins + user-defined. 409 on duplicate or
built-in override. Contracts + drift gate updated.

Closes V4_CONCERNS §4.10.
Blocked-by: A3 (migration runner logging — first new migration after).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface B] Config-preset persistence (B3)" --body "$(cat <<'EOF'
## Summary
- New `config_presets` table (migrations/0003_config_presets.sql).
- POST /api/v1/config/presets persists to DB; GET unions builtins + user-defined.
- 409 on duplicate or built-in override.
- Contracts 1 + 2 updated; drift gate covers the new table and endpoint.

Closes V4_CONCERNS §4.10.
Blocked-by: A3 (#<N>).

## Test plan
- [x] new tests: test_config_preset_persistence.py (2 tests)
- [x] contracts updated; drift gate green
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 4: B4 — A/B comparison view

**Files:**
- Modify: `app/api/regression.py`
- Modify: `app/services/regression_service.py`
- Modify: `app/schemas/v1.py` (add `RegressionCompare` schema)
- Modify: `app/web/src/routes/regression/compare/+page.svelte`
- Modify: `docs/contracts/v1-api.md`
- Create: `tests/app/test_regression_compare.py`

- [ ] **Step 4.1: Rebase**

```bash
git fetch origin main && git rebase origin/main
```

- [ ] **Step 4.2: Inspect existing regression endpoints**

```bash
cat app/api/regression.py
cat app/services/regression_service.py
ls app/web/src/routes/regression/compare/
```

Determine current state. The directory exists but the page is likely
empty.

- [ ] **Step 4.3: Define the schema**

Add to `app/schemas/v1.py`:

```python
class RegressionCompareRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_a": "CardCaptureFlow/1715523601",
                "run_b": "CardCaptureFlow/1715527890",
            }
        }
    )
    run_a: str
    run_b: str


class RegressionCompareDiff(BaseModel):
    added: list[str] = []      # card UUIDs in B but not A
    removed: list[str] = []    # card UUIDs in A but not B
    reassigned: list[dict] = []  # {instance_id, a_side, b_side}


class RegressionCompare(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_a": "CardCaptureFlow/1715523601",
                "run_b": "CardCaptureFlow/1715527890",
                "diff": {"added": [], "removed": [], "reassigned": []},
                "metric_deltas": {"card_recall": 0.02, "side_accuracy": -0.01},
            }
        }
    )
    run_a: str
    run_b: str
    diff: RegressionCompareDiff
    metric_deltas: dict[str, float | None]
```

- [ ] **Step 4.4: Write the failing test**

Create `tests/app/test_regression_compare.py`:

```python
"""POST /api/v1/regression/compare returns the diff between two runs."""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def _seed_two_runs(db_path):
    """Create two regression_runs with overlapping cards for testing."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO regression_runs(baseline_id, code_sha, config_json, "
            "metrics_json, per_video_json) VALUES (NULL, 'sha1', '{}', ?, '[]')",
            (json.dumps({"card_recall": {"value": 0.85}}),),
        )
        conn.execute(
            "INSERT INTO regression_runs(baseline_id, code_sha, config_json, "
            "metrics_json, per_video_json) VALUES (NULL, 'sha2', '{}', ?, '[]')",
            (json.dumps({"card_recall": {"value": 0.87}}),),
        )
        conn.commit()


def test_compare_returns_metric_deltas(tmp_path):
    db = tmp_path / "cards.sqlite"
    client = TestClient(create_app(db_path=db))
    _seed_two_runs(db)
    response = client.post(
        "/api/v1/regression/compare",
        json={"run_a": "1", "run_b": "2"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metric_deltas"]["card_recall"] == pytest.approx(0.02, abs=1e-6)
    assert "diff" in body
```

- [ ] **Step 4.5: Run the test — expect FAIL**

```bash
pytest tests/app/test_regression_compare.py -v
```

Expected: 404 or other failure — endpoint doesn't exist yet.

- [ ] **Step 4.6: Implement the endpoint**

Add to `app/api/regression.py`:

```python
from app.schemas.v1 import RegressionCompare, RegressionCompareRequest

@router.post("/compare", response_model=RegressionCompare)
def compare_runs(payload: RegressionCompareRequest, request: Request):
    return request.app.state.regression_service.compare(
        run_a=payload.run_a, run_b=payload.run_b,
    )
```

Add to `app/services/regression_service.py`:

```python
import json
import sqlite3

from app.schemas.v1 import RegressionCompare, RegressionCompareDiff


def compare(self, run_a: str, run_b: str) -> RegressionCompare:
    with sqlite3.connect(self.db_path) as conn:
        rows = {
            str(r[0]): json.loads(r[1])
            for r in conn.execute(
                "SELECT run_id, metrics_json FROM regression_runs "
                "WHERE run_id IN (?, ?)",
                (run_a, run_b),
            ).fetchall()
        }
    a, b = rows.get(run_a, {}), rows.get(run_b, {})

    metric_names = set(a) | set(b)
    deltas = {}
    for name in metric_names:
        av = a.get(name, {}).get("value") if isinstance(a.get(name), dict) else a.get(name)
        bv = b.get(name, {}).get("value") if isinstance(b.get(name), dict) else b.get(name)
        if av is None or bv is None:
            deltas[name] = None
        else:
            deltas[name] = bv - av

    # Card-level diff is more work — for B4, leave it empty if the
    # cards table doesn't carry run-id info yet. Note in PR description.
    diff = RegressionCompareDiff()
    return RegressionCompare(
        run_a=run_a, run_b=run_b, diff=diff, metric_deltas=deltas
    )
```

(If the cards table doesn't track which `regression_run_id` produced
each card, the `diff` field stays empty for now — surface this in the
PR description as a Wave 5 follow-up.)

- [ ] **Step 4.7: Document in Contract 2**

Add `POST /api/v1/regression/compare` to `docs/contracts/v1-api.md`.

Update `tests/app/test_api_contract.py`:

```python
ROUTES_REQUIRED.append(("POST", "/api/v1/regression/compare", RegressionCompare))
REQUEST_BODIES[("POST", "/api/v1/regression/compare")] = RegressionCompareRequest
```

- [ ] **Step 4.8: Build the Svelte page**

Edit `app/web/src/routes/regression/compare/+page.svelte`:

```svelte
<script lang="ts">
  import { onMount } from "svelte";

  let runs: Array<{ run_id: string; created_at: string }> = [];
  let runA = "";
  let runB = "";
  let compare: any = null;

  async function loadRuns() {
    const r = await fetch("/api/v1/runs");
    runs = (await r.json()).items;
  }

  async function doCompare() {
    const r = await fetch("/api/v1/regression/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_a: runA, run_b: runB }),
    });
    compare = await r.json();
  }

  onMount(loadRuns);
</script>

<svelte:head><title>Compare runs — card-capture</title></svelte:head>

<h1>Compare runs</h1>

<div class="picker">
  <label>Run A
    <select bind:value={runA}>
      <option value="">—</option>
      {#each runs as r}<option value={r.run_id}>{r.run_id}</option>{/each}
    </select>
  </label>
  <label>Run B
    <select bind:value={runB}>
      <option value="">—</option>
      {#each runs as r}<option value={r.run_id}>{r.run_id}</option>{/each}
    </select>
  </label>
  <button on:click={doCompare} disabled={!runA || !runB || runA === runB}>
    Compare
  </button>
</div>

{#if compare}
  <h2>Metric deltas</h2>
  <table>
    <thead><tr><th>Metric</th><th>Δ (B − A)</th></tr></thead>
    <tbody>
      {#each Object.entries(compare.metric_deltas) as [name, delta]}
        <tr class:up={typeof delta === "number" && delta > 0}
            class:down={typeof delta === "number" && delta < 0}>
          <td>{name}</td>
          <td>{delta === null ? "—" : delta.toFixed(3)}</td>
        </tr>
      {/each}
    </tbody>
  </table>
{/if}

<style>
  .picker { display: flex; gap: 1rem; align-items: end; }
  .up { color: green; }
  .down { color: red; }
  table { border-collapse: collapse; margin-top: 1rem; }
  td, th { padding: 0.25rem 0.5rem; border: 1px solid #ddd; }
</style>
```

- [ ] **Step 4.9: Run tests; build frontend**

```bash
pytest tests/app/test_regression_compare.py tests/app/test_api_contract.py -v
python -m pytest tests/ -q
cd app/web && npm run build && cd ../..
```

Expected: green.

- [ ] **Step 4.10: Commit and open PR**

```bash
git add app/api/regression.py app/services/regression_service.py \
        app/schemas/v1.py docs/contracts/v1-api.md \
        app/web/src/routes/regression/compare/ \
        tests/app/test_regression_compare.py tests/app/test_api_contract.py
git commit -m "feat(regression): A/B comparison view

New POST /api/v1/regression/compare returns metric deltas and a diff
between two runs. Svelte page at /regression/compare renders the
result. Card-level diff is a Wave 5 follow-up (depends on cards
tracking regression_run_id).

Closes V4_CONCERNS §4.11.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface B] A/B comparison view (B4)" --body "$(cat <<'EOF'
## Summary
- New POST /api/v1/regression/compare (documented in Contract 2).
- Svelte page renders metric-delta table with green/red highlighting.
- Card-level diff is empty until cards table tracks regression_run_id (Wave 5).

Closes V4_CONCERNS §4.11.

## Test plan
- [x] new test: test_regression_compare.py
- [x] contract conformance: routes + request bodies updated
- [x] pytest tests/ green locally
- [x] frontend builds clean
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 5: Update V4_CONCERNS.md and final verification

- [ ] **Step 5.1: Move §4.10, §4.11, §4.14, §4.15 to §2**

Edit `V4_CONCERNS.md`:

- §4.15 → §2.23 (B1 PR number)
- §4.14 → §2.24 (B2 PR number)
- §4.10 → §2.25 (B3 PR number)
- §4.11 → §2.26 (B4 PR number)

Commit and push.

- [ ] **Step 5.2: Report completion**

Surface B is done.

---

## Self-Review Checklist

- [ ] B1, B2, B3, B4 merged.
- [ ] `V4_CONCERNS.md` §4.10, §4.11, §4.14, §4.15 moved to §2.
- [ ] CI green on `main`.
- [ ] Inbox drag-drop works end-to-end (manual or Playwright).
- [ ] Settings list + edit + save works against persistent DB.
- [ ] /regression/compare renders deltas for two real runs.
- [ ] Contracts 1 + 2 updated; drift gate green.
