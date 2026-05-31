# V5.5 Completion — Phase B: Activate Static Enforcement

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every architecture/import check actually executable from a clean checkout and run on the PR lane. Specifically: ensure `.importlinter` matches the parent-plan spec exactly (an earlier attempt broadened `cv2.imgcodecs` to all of `cv2`, which is overreach), populate the empty `ci-lane-commands.md`, and commit the Phase B working-tree changes.

**Architecture:** Three tasks. `.importlinter`, `tests/architecture/test_import_linter.py`, and `.github/workflows/ci.yml` already exist in the working tree with mostly-correct content; this plan reconciles the one deviation, fills in the missing doc, and commits each piece atomically.

**Tech Stack:** Import Linter ≥2.0, pytest, GitHub Actions.

**Parent plan:** `docs/superpowers/plans/2026-05-28-v5-5-completion.md` (Phase B section). When this plan and the parent disagree, the parent wins.

**Acceptance:**
1. `.importlinter` has `include_external_packages = True` and the strict-gpu contract forbids `cv2.imgcodecs` (not all of `cv2`).
2. `tests/architecture/test_import_linter.py` runs on every default pytest invocation (no env-var gate) and fails loudly with an install hint when `lint-imports` is absent.
3. `.github/workflows/ci.yml` installs `[dev]` extras and runs the architecture lane + perf-smoke + `lint-imports` (blocking).
4. `docs/superpowers/plans/v5-5/ci-lane-commands.md` contains the canonical lane commands (currently it exists as a 0-byte file).
5. `lint-imports` exits successfully from a fresh `pip install -e ".[dev]"`. **Contract violations from raw `sqlite3` callers are EXPECTED and stay open until Phase C** — that violation set proves the linter is actually running and is the Phase C entry point.

**Pre-flight context (verified 2026-05-28):**
- `.importlinter` was rewritten and adds `include_external_packages = True` ✅, but the strict-gpu contract was broadened from `cv2.imgcodecs` (plan) to the entire `cv2` package — fix this.
- `tests/architecture/test_import_linter.py` was rewritten correctly ✅.
- `.github/workflows/ci.yml` was rewritten correctly ✅.
- `docs/superpowers/plans/v5-5/ci-lane-commands.md` exists but is **empty (0 B)** — populate it.
- Nothing is committed. All four files sit in the working tree.

---

## File Structure

```text
.importlinter                                          Modified; fix cv2 → cv2.imgcodecs
.github/workflows/ci.yml                               Modified (no edit, just commit)
tests/architecture/test_import_linter.py               Modified (no edit, just commit)
docs/superpowers/plans/v5-5/ci-lane-commands.md        Populated; currently empty
```

---

### Task B.1: Restore `cv2.imgcodecs` as the strict-gpu forbidden module (not all of `cv2`)

**Files:**
- Modify: `.importlinter`

The parent plan's `[importlinter:contract:strict-gpu-no-image-io]` contract forbids `PIL`, `PIL.Image`, and `cv2.imgcodecs`. The working tree forbids `PIL` and `cv2` (entire package). Broadening to `cv2` will cause false positives — strict-gpu code legitimately uses `cv2` for dtype constants and color conversions; the prohibition is specifically against the file-IO submodule `cv2.imgcodecs` (used for `cv2.imread`/`cv2.imwrite`).

- [ ] **Step 1: Read the current contract block**

Run:
```bash
grep -nA6 'strict-gpu-no-image-io' .importlinter
```

Expected output ends with `forbidden_modules =\n    PIL\n    cv2`. If it already lists `cv2.imgcodecs`, skip to Step 3 — the deviation was already fixed.

- [ ] **Step 2: Restore the original forbidden list**

Edit `.importlinter`. Find:

```ini
[importlinter:contract:strict-gpu-no-image-io]
name = strict GPU code must not import OpenCV/PIL file IO
type = forbidden
source_modules =
    card_capture.runtime.strict_gpu
forbidden_modules =
    PIL
    cv2
```

Replace the `forbidden_modules` block:

```ini
[importlinter:contract:strict-gpu-no-image-io]
name = strict GPU code must not import OpenCV/PIL file IO
type = forbidden
source_modules =
    card_capture.runtime.strict_gpu
forbidden_modules =
    PIL
    PIL.Image
    cv2.imgcodecs
```

Leave every other contract unchanged.

- [ ] **Step 3: Run the linter to confirm the strict-gpu contract still passes (or fails only for unrelated reasons)**

Run:
```bash
PYTHONPATH=src:. lint-imports 2>&1 | grep -A5 'strict-gpu-no-image-io'
```

Expected: the contract reports as KEPT (passing). If it BROKEN with a `cv2.imgcodecs` violation, the strict-gpu file actually imports `cv2.imgcodecs` and needs a separate fix — surface this and stop. If it BROKEN for `cv2` only, the deviation was not fully reversed.

- [ ] **Step 4: Commit**

Run:
```bash
git add .importlinter
git commit -m "fix(v55-phaseB): include_external_packages=True; strict-gpu forbids cv2.imgcodecs (not all of cv2)

The previous attempt broadened the strict-gpu forbidden list from
cv2.imgcodecs (file IO) to the whole cv2 package, which would forbid
legitimate dtype/colorspace uses inside strict_gpu.py. Restore the
narrower scope per docs/superpowers/specs/2026-05-24-v5-5-refactoring-design.md."
```

---

### Task B.2: Commit the already-correct `test_import_linter.py` and `ci.yml`

**Files:**
- Modify: `tests/architecture/test_import_linter.py` (commit only)
- Modify: `.github/workflows/ci.yml` (commit only)

These files were rewritten correctly by an earlier attempt but never committed.

- [ ] **Step 1: Diff the test file against the parent-plan specification**

Run:
```bash
diff <(sed -n '/Replace the file in full:/,/```$/p' docs/superpowers/plans/2026-05-28-v5-5-completion.md | awk '/```python/,/```$/' | sed '1d;$d' | head -80) tests/architecture/test_import_linter.py
```

If there's a semantically meaningful diff, align the file with the parent plan. A working-tree diff that adds `PYTHONPATH=src:.:` injection inside the test is acceptable (it's an improvement that makes the test work without external env setup).

- [ ] **Step 2: Verify the test runs (failure is OK at this point)**

Run:
```bash
unset V55_IMPORT_LINT_BLOCKING
python3 -m pytest tests/architecture/test_import_linter.py -v
```

Expected outcome — exactly one of:
- PASS (if all contracts are satisfied — unlikely until Phase C completes).
- FAIL with a body listing the `lint-imports` stderr output (showing the raw-sqlite3 contract violations Phase C will close). This is the intended state and proves the linter is wired.

If the test errors with `lint-imports binary not found on PATH`, install the dev extras:
```bash
python3 -m pip install -e ".[dev]"
export PATH="$(python3 -m site --user-base)/bin:$PATH"
```
…and re-run.

- [ ] **Step 3: Validate `.github/workflows/ci.yml` parses**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "yaml ok"
```

Expected: `yaml ok`.

- [ ] **Step 4: Commit the test file**

Run:
```bash
git add tests/architecture/test_import_linter.py
git commit -m "test(v55-phaseB): Import Linter test always runs; install hint on missing binary

Phase C is expected to close the remaining raw-sqlite3 contract violations;
this commit only makes them visible to CI."
```

- [ ] **Step 5: Commit the CI workflow**

Run:
```bash
git add .github/workflows/ci.yml
git commit -m "ci(v55-phaseB): install [dev] extras; run architecture lane + lint-imports

Adds a separate 'Architecture lane' step that runs tests/architecture/,
a perf-smoke step, and a blocking lint-imports step. PYTHONPATH is set
to src:. so Import Linter can resolve card_capture without an install hook."
```

---

### Task B.3: Populate `docs/superpowers/plans/v5-5/ci-lane-commands.md`

**Files:**
- Modify: `docs/superpowers/plans/v5-5/ci-lane-commands.md` (currently empty)

- [ ] **Step 1: Confirm the file is empty**

Run:
```bash
wc -c docs/superpowers/plans/v5-5/ci-lane-commands.md
```

Expected: `0`. If non-zero, read it and decide whether to overwrite or merge; do not blindly replace existing content.

- [ ] **Step 2: Write the canonical commands**

Write `docs/superpowers/plans/v5-5/ci-lane-commands.md`:

```markdown
# V5.5 CI Lane Commands

## One-time setup

```bash
python3 -m pip install -e ".[dev]"
# If lint-imports is not on PATH, add user-site bin:
export PATH="$(python3 -m site --user-base)/bin:$PATH"
```

## Fast PR lane (default; no GPU, no credentials, no real videos)

```bash
python3 -m pytest tests/ -q
python3 -m pytest tests/architecture/ -q
python3 -m pytest tests/performance/test_perf_harness_smoke.py -q
PYTHONPATH=src:. lint-imports
```

The `addopts` line in `pyproject.toml` already deselects `quarantine` and `benchmark` markers.

## Optional hardware lane (CUDA)

```bash
python3 -m pytest tests/ -q -m cuda
```

## Optional hardware lane (MPS)

```bash
python3 -m pytest tests/ -q -m mps
```

## Provider lane (requires credentials)

```bash
python3 -m pytest tests/ -q -m provider
```
```

- [ ] **Step 3: Confirm non-empty**

Run:
```bash
wc -l docs/superpowers/plans/v5-5/ci-lane-commands.md
```

Expected: ~30 lines.

- [ ] **Step 4: Commit**

Run:
```bash
git add docs/superpowers/plans/v5-5/ci-lane-commands.md
git commit -m "docs(v55-phaseB): document fast PR lane commands"
```

**Phase B complete.** Import Linter is wired and running, the architecture lane is in CI, and the canonical commands are documented. Phase C's caller migration is the next blocker for green CI.
