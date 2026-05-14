# v4 Concerns — Pass 2 Review

**Owner:** Josh (jpglick)
**Reviewer pass:** 2026-05-13
**Scope:** Verify resolution claims in `V4_CONCERNS.md` and flag anything
new uncovered while doing so. Companion to `V4_CONCERNS.md`; does not
supersede it.

**TL;DR**
- Most of the §1.x concerns from `V4_CONCERNS.md` are genuinely closed.
- A few are *softly* closed (deprecation flag rather than deletion;
  bridge function rather than consolidation) and should be re-classified
  as new concerns below.
- Ten new concerns surfaced during the verification pass, several of
  which are silent-failure modes that won't trip a test but will rot
  the system if left.
- Plan items §4.1–§4.9, §4.11, §4.12, §4.13 from `V4_CONCERNS.md` are
  still genuinely unimplemented and are not duplicated below.

Severity tags follow the convention in `V4_CONCERNS.md`.

---

## 1. Verification of prior concerns

| Prior ID | Status | Evidence |
|---|---|---|
| §1.1 monolith vs Metaflow | **Soft-resolved** — see new §2.1 | `cli.py:45-49` deprecation flag; `tests/pipeline/test_path_equivalence.py` exists; monolith file still 2,243 lines |
| §1.2 uncommitted work | **Resolved** | `git status` clean |
| §1.3 golden-set bootstrap | **Resolved** | `golden_set/` tracked; `tests/fixtures/golden_corpus/IMG_5872/` has both `.MOV` and `.truth.json` |
| §1.4 `harness_config.json` untracked | **Resolved** | File removed from root; `harness/config.example.json` documents the schema |
| §1.5 `FBPredictor` random labels | **Resolved** | `ml/inference/fb_predict.py` raises `UntrainedModelError`; `is_available()` classmethod added |
| §1.6 reid_embedding NULL on ByteTrack | **Resolved** | `pipeline/steps/store.py:62-67` always computes a DINOv2 embedding on the fused image when the tracker didn't produce one |
| §1.7 no frozen baseline | **Deferred** per user |
| §1.8 migration silent skip | **Resolved** | `migrations/run_migrations.py` prints "Applied" / "Skipped (table missing, will retry)" |
| §1.9 no CI gate | **Resolved** | `.github/workflows/ci.yml` runs `pytest tests/` + `scripts/validate_schema_docs.py` on push/PR to main |
| §1.10 schema drift | **Soft-resolved** — see new §2.6 | Regex-based MD parser in `scripts/validate_schema_docs.py` |
| §1.11 mixed float/dataclass metrics | **Resolved** | `harness/metrics/types.py` defines `MetricResult` (Pydantic, frozen); all 5 metrics return it; `runner.py` types `metrics: dict[str, MetricResult]` |
| §1.12 `config={}` TODO in harness CLI | **Resolved** | No `TODO` strings remain in `harness/cli.py` |
| §1.13 truth-file naming convention | **Resolved** | `harness/runner.py:_find_truth` picks `<video_id>.truth.json` as canonical and emits `DeprecationWarning` on the other two |
| §1.14 synthetic eval | **Deferred** (no real F/B training data yet) |
| §1.15 `app/web/node_modules` size | Unchanged — still gitignored, no `npm ci` policy documented |
| §1.16 D1 PR retrospective note | Unchanged — paperwork only |
| §4.10 user preset persistence | **Resolved** | `migrations/0003_config_presets.sql` adds the table; `app/api/config.py` reads/writes it with 409 on built-in collision |
| §4.14 Settings tab | **Partial** — see new §2.7 | `/settings` page exists but only edits 5 thresholds, not the full knob set |
| §4.15 drag-drop inbox | **Partial** — see new §2.3 | UI accepts drops; only the filename is sent to the backend, not the file bytes |
| §4.16 two competing config dataclasses | **Soft-resolved** — see new §2.2 | `PipelineConfig.to_options()` bridges to `ProcessingOptions`; both dataclasses still defined separately |
| §4.17 15-labeled-videos | **Deferred** per user |

---

## 2. New concerns

### 2.1 — Monolith pipeline still maintained alongside Metaflow — **High**

**Where:** `src/card_capture/pipeline.py` (2,243 lines, up from
~2,079 at original §1.1 reporting); `cli.py:45-49` still accepts
`--pipeline monolith`; `tests/test_cli.py` lines 92, 130, 167, 213 all
hit `"--pipeline", "monolith"` as the *primary* CLI test path.

**Why it's not resolved by the deprecation warning:** the original §1.1
ask was to "delete or freeze the monolith." Adding a help-text
deprecation note and writing an equivalence test does not freeze
anything — both code paths are still maintained, both can still receive
bug fixes that drift apart, and the canonical CLI smoke tests still
exercise the *deprecated* path rather than the *canonical* one. The
risk model from the original §1.1 is unchanged.

Compounding: `test_path_equivalence.py:78-85` is intentionally weak —
it compares only the set of `angle` values (`{Front, Back}`) plus
total count, explicitly ignoring `instance_id` and `visual_hash`
because `FakeCardDetector` adds random noise. So a regression where
the monolith drops 3 real cards and the Metaflow drops 4 different
real cards but both end up with the same `{Front, Back}` set and
the same count will not fail this test.

**Fix:** Pick a wave for monolith deletion (the deprecation message
says "Wave 5") and write a deletion checklist now:
1. Migrate `tests/test_cli.py` to default to `--pipeline metaflow`
   (or omit the flag) at all four call sites.
2. Strengthen `test_path_equivalence` to compare `(angle, session_id,
   frame_index_bucket)` triples rather than just the angle set, so the
   test detects real card-set divergence.
3. Schedule the deletion in a milestone, not just a help string.

---

### 2.2 — `PipelineConfig.to_options()` silently drops fields — **High**

**Where:** `src/card_capture/config.py:to_options()` (lines roughly
60-95) maps `PipelineConfig` → `pipeline.ProcessingOptions` by listing
field names one-by-one.

**The drop:** `PipelineConfig` has these fields that are **not**
mapped by `to_options()`:
- `detector`
- `fast_scan_fps`
- `confirm_scan_fps`
- `valley_drop_ratio`
- `valley_min_width_frames`
- `delta_spike_ratio`
- `reid_distance_threshold`
- `fusion_target_frames`
- `corner_refinement`

Some of these (e.g. `valley_drop_ratio`) don't exist on
`ProcessingOptions` at all — meaning the monolith genuinely *cannot*
honor them even if the user sets them in config. The user setting will
be silently ignored on the monolith path and silently honored on the
Metaflow path: this is the exact failure mode §1.4 / §4.16 warned
about, just relocated from "field defaults disagree" to "fields
present on one path but not the other."

**Fix:** Either (a) make `ProcessingOptions` a true subset of
`PipelineConfig` and generate `to_options()` programmatically (e.g.
`ProcessingOptions(**{k: v for k, v in cfg.__dict__.items() if k in
processing_field_names})`), or (b) delete `ProcessingOptions` and have
the monolith accept `PipelineConfig` directly. Option (b) is preferred
because it eliminates the duplicate dataclass entirely; option (a)
keeps the duplication but at least makes drift impossible.

Either fix should be accompanied by a test that asserts
`set(PipelineConfig fields) - set(ProcessingOptions fields)` is empty
(or that the difference is an explicit, named whitelist).

---

### 2.3 — Drag-drop video upload posts only the filename, not the file bytes — **High**

**Where:** `app/web/src/routes/videos/+page.svelte:onDrop()` and
`onFileInput()`. Both call:

```ts
async function registerVideo(filename: string) {
    await api.videos.create({ filename });
}
```

The actual `File` object is discarded. The backend receives only a
filename string and presumably looks for the file in a known directory
on disk.

**Why it matters:** dropping a video from `~/Desktop/foo.mov` onto the
web UI will create a video record with `filename="foo.mov"` and no
corresponding file. The pipeline will run against a path that doesn't
exist and either crash or skip silently depending on which error path
fires.

The plan (Appendix A.4.1) explicitly says "drag-drop video upload."
The current UI fakes the upload — it accepts the drop event but doesn't
upload the bytes.

**Fix:** Either (a) implement multipart upload (`POST /api/v1/videos`
with `UploadFile`), persist into a managed `videos/` directory, and
return the new record; or (b) make the drag-drop UX honest by
explicitly requiring a path under a watched directory and rejecting
drops from elsewhere with a helpful error. Option (a) is what the plan
intends.

---

### 2.4 — `pipeline/steps/store.py` swallows all exceptions during late ReID embedding — **Medium**

**Where:** `pipeline/steps/store.py:62-69`.

```python
# Task C2: Always populate reid_embedding even if tracker didn't.
# We use the FUSED image for the stable embedding.
try:
    from card_capture.ml.embeddings import compute_reid_embedding
    emb = compute_reid_embedding(f["fused_image_path"])
    embedding_bytes = emb.tobytes()
except Exception as e:
    print(f"Failed to generate late ReID embedding for {iid[:8]}: {e}")
```

If `compute_reid_embedding` fails — DINOv2 weights missing, GPU OOM,
file path corrupt — every record from that run lands with
`reid_embedding = NULL` and the only signal is a stdout line. The
column will look populated on some rows and NULL on others, with no
telemetry, no `pipeline_events` row, no `run_telemetry.json` field
recording the failure rate.

This is the same failure mode as the *original* §1.6 (silent NULL),
just relocated from "the tracker didn't run" to "the embedder
crashed."

**Fix:** Catch the specific exceptions you expect (`FileNotFoundError`,
`torch.cuda.OutOfMemoryError`, `RuntimeError`) and record a
`pipeline_events` row with reason `reid_embedding_failed` and the
exception class. For unexpected exceptions, re-raise; a crashed model
load should not be a silent run-time NULL.

---

### 2.5 — Migration runner splits SQL on `;` naively — **Medium**

**Where:** `migrations/run_migrations.py:_split_statements()`.

```python
def _split_statements(sql: str) -> list:
    return [s.strip() for s in sql.split(";") if s.strip()]
```

The docstring acknowledges this is naive: "sufficient for migration
files whose SQL does not contain semicolons inside string literals or
block comments." The current three migration files happen not to hit
the limitation, but the next migration that adds a trigger
(`BEGIN ... END;`), a view definition with a semicolon inside a string
literal, or a stored function will break silently — `sqlite3` will see
mangled fragments and either error on a subsequent statement or
silently apply a partial DDL.

**Fix:** Use `conn.executescript(sql)` for migration files and let
SQLite parse the statements properly. The current loop-and-catch
`OperationalError` pattern can be replaced with: try
`executescript()`, catch known idempotency errors via the resulting
exception's message, and otherwise let it propagate. Or, if statement-
level error handling is genuinely needed, use a real SQL tokenizer
(`sqlglot`, `sqlparse`) rather than a `.split(";")`.

---

### 2.6 — Schema-drift validator is a regex MD parser — **Medium**

**Where:** `scripts/validate_schema_docs.py`.

The script self-documents as "Very simple parser" and uses two regexes
to extract `CREATE TABLE` and `ALTER TABLE` blocks from both SQL files
and a markdown contract doc, then diffs the column sets. Failure modes:
- Markdown table reformatting (added/removed blank lines, different
  fence styles) silently changes what the regex matches.
- Quoted identifiers, multi-line `CHECK` constraints, or column-level
  comments containing the keyword `COLUMN` will confuse the regex.
- The diff is set-based, so a column rename that adds the new name
  but doesn't remove the old will pass.

`V4_CONCERNS.md §1.10` proposed two fix candidates and called the
"generate the contract markdown from the Pydantic schema" path
"lowest friction." The implemented path is the *other* fix candidate
(parse both sides and diff), implemented at the floor of robustness.

**Fix:** Either (a) move to schema-generated docs as originally
proposed — single source of truth, no parser needed — or (b) replace
the regex with `sqlglot.parse()` for the SQL side and a JSON Schema
extraction (Pydantic models → `model_json_schema()`) for the contract
side.

---

### 2.7 — Settings page covers 5 thresholds out of ~30 — **Medium**

**Where:** `app/web/src/routes/settings/+page.svelte` — the
`THRESHOLDS` map in the script tag exposes:

- `corner_confidence`
- `background_novelty_threshold`
- `centroid_jump_ratio`
- `valley_drop_ratio`
- `foil_threshold`

The full `PipelineConfig` has 30+ tunable fields (e.g.
`min_track_length`, `fast_scan_fps`, `triage_keep_percentile`,
`fusion_target_frames`, `reid_distance_threshold`,
`delta_spike_ratio`, etc.). The Settings page also exposes only one
named view: the preset editor. The plan (Appendix A.4.7) calls for
"threshold sliders with tooltips explaining trade-offs" — for *every*
preset-editable threshold.

This is OK as a first cut but should be tracked so it doesn't become
"the settings page" by accident.

**Fix:** Drive the threshold list off `PipelineConfig` introspection
rather than a hand-listed dict, so adding a field automatically adds
the slider (with a `# tooltip: ...` annotation harvested from a
field metadata source). At minimum, document which thresholds are
intentionally hidden from the UI and why.

---

### 2.8 — Path-equivalence test is non-deterministic and weak — **Medium**

**Where:** `tests/pipeline/test_path_equivalence.py`.

Two weaknesses:
1. Uses `FakeCardDetector`, which adds random noise to confidence per
   call. Re-running the test produces different `primary_hash` values
   even on the same path. The test acknowledges this and works around
   it by comparing only `angle` sets — but that workaround also throws
   away the ability to detect real card-set divergence (see new §2.1).
2. The test is `@pytest.mark.skipif(not FIXTURE_VIDEO.exists())`, but
   `tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV` *does* exist
   in the repo, so the test will actually run in CI. That's good — but
   running CI against a real MOV via `FakeCardDetector` produces a flaky
   signal because the random seed isn't pinned.

**Fix:** (a) seed `FakeCardDetector` deterministically (env var or
constructor arg) so the test is reproducible; (b) compare on
`(angle, session_id)` pairs and total count, not just `angle` sets;
(c) once the monolith is removed (per new §2.1), delete this test.

---

### 2.9 — Config-preset table presence is silently optional — **Medium**

**Where:** `app/api/config.py:_get_user_presets()`.

```python
try:
    with sqlite3.connect(...) as conn:
        ...
except sqlite3.OperationalError:
    # Table may not exist yet if migration hasn't run.
    return []
```

If `migrations/0003_config_presets.sql` failed to apply (e.g. the
`run_migrations.py` "no such table, will retry" silent-skip path
fires on an unrelated upstream migration), `_get_user_presets` returns
`[]` and the UI silently shows only built-in presets. The user
clicks "+ New Preset", saves it, gets a 500 ("no such table"), and
has to dig into logs to understand why.

`V4_CONCERNS.md §2.2` confirms `apply_migrations(db_path)` runs at
app startup. But the *result* of that call isn't surfaced anywhere —
if a migration partially-skipped per the §1.8 path, the app comes up
healthy and the missing table only surfaces on first write.

**Fix:** At app startup, assert that the *latest* expected migration
filename appears in `_migrations`; if not, log loudly (or refuse to
serve preset endpoints) until the operator runs migrations explicitly.

---

### 2.10 — `ProcessingOptions.rotate_180` default flipped from `False` to `True` — **Low**

**Where:** `src/card_capture/pipeline.py:213` (ProcessingOptions):
`rotate_180: bool = True`.

`CLAUDE.md §4` documents this as `rotate_180 = False` ("camera mounted
upside-down flag"). The default was flipped to `True` somewhere along
the way and `CLAUDE.md` was not updated. `PipelineConfig.rotate_180`
in `config.py:32` defaults to `False`. So:
- Programs that construct `ProcessingOptions()` directly get
  `rotate_180=True`.
- Programs that construct `PipelineConfig()` and pass through
  `.to_options(out)` get `rotate_180=False` (the value from
  `PipelineConfig`).

The two entry points produce different default orientations on the
same input video. This is the same shape of drift as §1.4 / new §2.2,
just for a single field.

**Fix:** Pick one. Recommend `False` to match `PipelineConfig` and
`CLAUDE.md`. Add a test that asserts
`PipelineConfig().to_options(Path("/tmp")).rotate_180 ==
PipelineConfig().rotate_180`.

---

## 3. Plan items still open

These remain unresolved exactly as documented in `V4_CONCERNS.md §4`
and are not re-listed in detail here:

- §4.1 Learned quality ranker — **High**
- §4.2 Per-pixel background variance — **High**
- §4.3 Per-region detector confidence — **High**
- §4.4 Content-aware F/B / dedup metric — **High** *(blocked on F/B
  classifier training data)*
- §4.5 Higher-resolution canvas (1000×1400) + Lanczos — **Medium**
- §4.6 YOLO26-OBB swap on CoreML — **Medium**
- §4.7 VideoToolbox decoder — **Medium**
- §4.8 Detection-conditioned sampler — **Medium**
- §4.9 Multi-process structured error codes — **Medium**
- §4.11 A/B compare view body (route exists, end-to-end unverified) — **Medium**
- §4.12 Sampler retrain path — **Low**
- §4.13 Real ReID on BoT-SORT — **Low** *(partly sidestepped by DINOv2
  fallback in `store.py`)*
- §4.17 15 labeled videos + frozen baseline — **[deferred per user]**

---

## 4. Recommended next-cycle ordering

Honest priority for the next dev cycle, based on this review:

1. **§2.3 (drag-drop only sends filename)** — user-visible bug; a
   one-screen feature that doesn't work end-to-end.
2. **§2.2 (`to_options` silently drops fields)** and **§2.10
   (`rotate_180` default flipped)** — same root cause; fix together by
   either deleting `ProcessingOptions` or generating it from
   `PipelineConfig`. Forecloses an entire class of silent drift.
3. **§2.1 (monolith still maintained)** — schedule deletion in a
   named wave with a checklist. Pulls 2,243 lines of duplicate code
   out of the maintenance surface and lets §2.8 (the weak parity test)
   go with it.
4. **§2.4 (`store.py` silent exception swallow)** — replace with a
   typed catch and a `pipeline_events` row. One commit.
5. **§2.9 (preset-table silently optional)** — small startup
   assertion. One commit.
6. **§2.5 (naive `;` split)** and **§2.6 (regex MD parser)** —
   robustness work on the infrastructure that everything else
   depends on. Bundle into one "infra hardening" commit.
7. **§2.7 (Settings page only covers 5 thresholds)** and **§2.8
   (path-equivalence test weakness)** — polish; ship when convenient.

Plan items §4.1, §4.2, §4.3, §4.4 from `V4_CONCERNS.md` remain the
high-leverage *algorithmic* work blocked on training data and explicit
scope, and are not duplicated here.

---

## 5. Notes for the next reviewer

- `V4_CONCERNS.md` is the long-form living document. This file is a
  point-in-time review pass. When concerns here resolve, prefer
  promoting their resolution lines into `V4_CONCERNS.md §2` rather
  than maintaining two parallel registers.
- The pattern of "soft-resolved" concerns (deprecation message instead
  of deletion, bridge function instead of consolidation) appears
  three times in this pass (§1.1 → §2.1, §1.10 → §2.6, §4.16 →
  §2.2). Worth watching for as a review smell on subsequent waves.
- Working tree was clean at review time, CI is green, harness tests
  exist and use `MetricResult` end-to-end. Foundation is solid; the
  concerns above are about *finish* and *long-tail correctness*, not
  about the core build being broken.
