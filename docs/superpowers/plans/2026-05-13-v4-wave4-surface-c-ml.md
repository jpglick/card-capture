# Wave 4 — Surface C (ML) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the F/B classifier from silently emitting random predictions when untrained, and ensure the `reid_embedding` column is populated regardless of tracker backend.

**Architecture:** Single agent, ~2 PRs. Surface C owns ML inference, dedup, and tracking. Blocked-by Surface E (CI). Independent of A/B/D *except* the C2 hook touches `pipeline/steps/store.py` — coordinate with A.

**Tech Stack:** PyTorch, ResNet-18, DINOv2 (ViT-S/14), FAISS, Python 3.11.

**Spec:** `docs/superpowers/specs/2026-05-13-v4-wave4-hardening-design.md` §6.

**Files owned by Surface C:** `src/card_capture/ml/**`, `src/card_capture/deduplicator.py`, `src/card_capture/tracking/**`, `pipeline/steps/dedup.py`. Single targeted edit to `pipeline/steps/store.py` (A owns it; rebase on A1 before opening C2).

---

## Pre-flight

- [ ] **P1: Confirm E1 is merged**

```bash
git fetch origin main
ls .github/workflows/test.yml
```

Expected: file exists on `main`.

- [ ] **P2: Create the worktree**

```bash
git worktree add ../card-capture-wave4-c -b wave4/c-ml origin/main
cd ../card-capture-wave4-c
pip install -e ".[harness,test]"
python -m pytest tests/ -q
```

Expected: tests pass.

---

## Task 1: C1 — `FBPredictor` refuses without checkpoint

**Files:**
- Create: `src/card_capture/ml/errors.py`
- Modify: `src/card_capture/ml/inference/fb_predict.py`
- Create: `tests/ml/test_fb_predict.py`
- Modify: every callsite of `FBPredictor`

- [ ] **Step 1.1: Inventory FBPredictor callsites**

```bash
grep -rn "FBPredictor" src/ app/ pipeline/ harness/ 2>/dev/null
```

Note every callsite. Each one will need an `is_available` guard added in
Step 1.5.

- [ ] **Step 1.2: Create the error type**

Create `src/card_capture/ml/errors.py`:

```python
"""ML error types."""
from __future__ import annotations


class UntrainedModelError(RuntimeError):
    """Raised when a model is asked to predict without a trained checkpoint."""
```

- [ ] **Step 1.3: Write the failing tests**

Create `tests/ml/test_fb_predict.py`:

```python
"""Tests for FBPredictor refusing predictions without a checkpoint.

Closes V4_CONCERNS §1.5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from card_capture.ml.errors import UntrainedModelError
from card_capture.ml.inference.fb_predict import FBPredictor


def test_predictor_refuses_when_no_checkpoint_path():
    with pytest.raises(UntrainedModelError):
        FBPredictor(checkpoint_path=None)


def test_predictor_refuses_when_checkpoint_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(UntrainedModelError):
        FBPredictor(checkpoint_path=missing)


def test_is_available_returns_false_when_no_checkpoint(tmp_path: Path):
    assert FBPredictor.is_available(None) is False
    assert FBPredictor.is_available(tmp_path / "missing.pt") is False


def test_is_available_returns_true_when_checkpoint_exists(tmp_path: Path):
    # Write any non-empty file at the path; we're testing the presence
    # check, not the load.
    fake_ckpt = tmp_path / "fake.pt"
    fake_ckpt.write_bytes(b"x")
    assert FBPredictor.is_available(fake_ckpt) is True


def test_predictor_loads_and_predicts_with_real_checkpoint(tmp_path: Path):
    """When given a real checkpoint, the predictor loads and returns
    a (label, confidence) tuple."""
    import torch
    from card_capture.ml.fb_classifier import FBClassifier

    # Save a freshly-constructed model as a checkpoint — its
    # predictions will be near-random, but the API contract must hold.
    ckpt = tmp_path / "fb.pt"
    torch.save(FBClassifier(pretrained=False).state_dict(), ckpt)

    pred = FBPredictor(checkpoint_path=ckpt)
    img = np.zeros((1050, 750, 3), dtype=np.uint8)  # arbitrary BGR
    label, conf = pred.predict(img)

    assert label in ("front", "back")
    assert 0.0 <= conf <= 1.0
```

- [ ] **Step 1.4: Run the tests — expect FAIL**

```bash
pytest tests/ml/test_fb_predict.py -v
```

Expected: at least `test_predictor_refuses_when_no_checkpoint_path` and
`test_predictor_refuses_when_checkpoint_missing` fail (predictor still
silently accepts `None`).

- [ ] **Step 1.5: Implement the refusal**

Edit `src/card_capture/ml/inference/fb_predict.py`:

```python
"""Inference wrapper for the Front/Back side classifier.

This predictor REQUIRES a trained checkpoint. With no checkpoint, the
underlying ResNet-18 has a randomly-initialised classification head and
predictions are random; rather than emit confident garbage we refuse.

Callers should guard with `FBPredictor.is_available(path)` and fall
back to the longest-track heuristic when the predictor is unavailable.

Closes V4_CONCERNS §1.5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import cv2
import numpy as np
from PIL import Image

from ..errors import UntrainedModelError
from ..fb_classifier import FBClassifier, get_transforms
from ..scaffolding import pick_device


class FBPredictor:
    def __init__(self, checkpoint_path: str | Path | None):
        if not self.is_available(checkpoint_path):
            raise UntrainedModelError(
                f"FBPredictor requires a trained checkpoint; got "
                f"{checkpoint_path!r}. Use FBPredictor.is_available() "
                f"before instantiation to fall back to the longest-track "
                f"heuristic."
            )

        self.device = pick_device()
        self.model = FBClassifier(pretrained=False)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.to(self.device)
        self.model.eval()
        self.transform = get_transforms()

    @classmethod
    def is_available(cls, checkpoint_path: str | Path | None) -> bool:
        if checkpoint_path is None:
            return False
        return Path(checkpoint_path).exists()

    @torch.no_grad()
    def predict(
        self, image: np.ndarray | Image.Image | str | Path
    ) -> Tuple[str, float]:
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image.convert("RGB")

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        label = "front" if idx.item() == 0 else "back"
        return label, float(conf.item())
```

- [ ] **Step 1.6: Update every callsite with an `is_available` guard**

For each line found in Step 1.1, wrap the construction:

```python
# BEFORE:
predictor = FBPredictor()  # or FBPredictor(checkpoint_path=None)
label = predictor.predict(image)

# AFTER:
from card_capture.ml.inference.fb_predict import FBPredictor
ckpt_path = Path("models/fb_classifier.pt")  # or wherever the config points
if FBPredictor.is_available(ckpt_path):
    predictor = FBPredictor(checkpoint_path=ckpt_path)
    label, conf = predictor.predict(image)
else:
    # Fall back to the longest-track heuristic. The pipeline's existing
    # angle/side assignment logic continues to apply.
    label = None  # signals "use the heuristic"
```

If a callsite already has heuristic-based fallback, leave it as-is and
just gate the predictor construction. If no callsite of `FBPredictor`
exists yet (it's wired but not used), that's fine — record that fact in
the PR description.

At application startup (`app/main.py:create_app` or the Metaflow
pipeline's `start` step), log once:

```python
from card_capture.ml.inference.fb_predict import FBPredictor

ckpt = Path("models/fb_classifier.pt")
if not FBPredictor.is_available(ckpt):
    log.warning(
        "F/B classifier checkpoint not found at %s; falling back to "
        "longest-track heuristic. Train a checkpoint to enable the "
        "learned side prediction.",
        ckpt,
    )
```

- [ ] **Step 1.7: Run the tests — expect PASS**

```bash
pytest tests/ml/test_fb_predict.py -v
python -m pytest tests/ -q
```

Expected: both green.

- [ ] **Step 1.8: Commit and open PR**

```bash
git add src/card_capture/ml/errors.py \
        src/card_capture/ml/inference/fb_predict.py \
        tests/ml/test_fb_predict.py \
        $(grep -rl "FBPredictor" src/ app/ pipeline/ 2>/dev/null)
git commit -m "fix(ml): FBPredictor refuses to predict without a trained checkpoint

Without a checkpoint, the underlying ResNet-18 has a randomly-initialised
2-class head and predictions are random. Now raises UntrainedModelError
on construction; callers gate with FBPredictor.is_available() and fall
back to the longest-track heuristic.

Closes V4_CONCERNS §1.5.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push -u origin wave4/c-ml
gh pr create --title "[Wave 4 — Surface C] FBPredictor refuses without checkpoint (C1)" --body "$(cat <<'EOF'
## Summary
- FBPredictor raises UntrainedModelError without a checkpoint.
- Adds FBPredictor.is_available() for callers to probe.
- All callsites guarded; startup logs a one-line warning when the checkpoint is missing.

Closes V4_CONCERNS §1.5.

## Test plan
- [x] new tests added: tests/ml/test_fb_predict.py (5 tests)
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 2: C2 — `reid_embedding` policy

**Files:**
- Create: `src/card_capture/ml/embeddings.py`
- Modify: `src/card_capture/deduplicator.py` (extract DINOv2 call)
- Modify: `pipeline/steps/store.py` (call the new helper) — **rebase on A1 first**
- Create: `tests/ml/test_embeddings.py`
- Create: `tests/pipeline/test_reid_embedding_populated.py`

- [ ] **Step 2.1: Update from main; rebase on A1**

```bash
git fetch origin main && git rebase origin/main
```

If A1 hasn't merged yet, wait. C2 depends on the structure of
`pipeline/steps/store.py` after A1.

- [ ] **Step 2.2: Investigate the existing DINOv2 wiring**

```bash
grep -n "dino\|DINOv2\|faiss" src/card_capture/deduplicator.py
```

Determine: (a) is DINOv2 already importable as a module-level helper,
or (b) is the embedding computation inline inside `deduplicator.py`'s
class methods? If (b), extracting it is the first job.

**Escalation check:** If DINOv2 isn't actually a callable module
(e.g. it's a wrapped HuggingFace pipeline embedded deep in another
function), the scope of C2 grows. Stop and report to the user before
forging ahead with a refactor.

- [ ] **Step 2.3: Write the failing tests**

Create `tests/ml/test_embeddings.py`:

```python
"""Tests for the reusable embedding helper.

Closes V4_CONCERNS §1.6.
"""
from __future__ import annotations

import numpy as np
import pytest

from card_capture.ml.embeddings import compute_reid_embedding


def test_compute_reid_embedding_returns_float32_vector():
    img = np.random.randint(0, 256, size=(1050, 750, 3), dtype=np.uint8)
    vec = compute_reid_embedding(img)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.ndim == 1
    # DINOv2 ViT-S/14 is 384-dim; allow other dims if the project changes
    # backbone, but assert non-trivial.
    assert vec.size >= 64


def test_compute_reid_embedding_is_deterministic():
    img = np.random.RandomState(0).randint(
        0, 256, size=(1050, 750, 3), dtype=np.uint8
    )
    vec1 = compute_reid_embedding(img)
    vec2 = compute_reid_embedding(img)
    np.testing.assert_allclose(vec1, vec2, rtol=1e-5)
```

Create `tests/pipeline/test_reid_embedding_populated.py`:

```python
"""Integration: after a full pipeline run, every card_instances row has
a non-null reid_embedding regardless of tracker backend.

Closes V4_CONCERNS §1.6.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_VIDEO = Path("tests/fixtures/fake_video.mov")


@pytest.mark.skipif(
    not FIXTURE_VIDEO.exists(),
    reason="fixture video not present",
)
@pytest.mark.parametrize("tracker", ["bytetrack", "botsort"])
def test_reid_embedding_populated_for_every_card(tmp_path, tracker):
    out = tmp_path / tracker
    subprocess.run(
        [
            sys.executable, "-m", "card_capture.cli", "process",
            str(FIXTURE_VIDEO),
            "--output-dir", str(out),
            "--db", str(out / "cards.sqlite"),
            "--detector", "fake",
            "--tracker-backend", tracker,
        ],
        check=True,
    )

    with sqlite3.connect(out / "cards.sqlite") as conn:
        rows = conn.execute(
            "SELECT instance_id, reid_embedding "
            "FROM card_instances WHERE reid_embedding IS NULL"
        ).fetchall()

    assert not rows, (
        f"{tracker}: {len(rows)} card_instances have NULL reid_embedding: "
        f"{[r[0] for r in rows]}. "
        f"§1.6 policy: column must be populated regardless of tracker."
    )
```

- [ ] **Step 2.4: Run the tests — expect FAIL**

```bash
pytest tests/ml/test_embeddings.py tests/pipeline/test_reid_embedding_populated.py -v
```

Expected: `test_embeddings.py` fails on `ModuleNotFoundError` (the
module doesn't exist yet). Integration test fails or skips depending on
fixture availability.

- [ ] **Step 2.5: Create the helper module**

Create `src/card_capture/ml/embeddings.py`:

```python
"""Reusable embedding helpers.

`compute_reid_embedding` produces a fixed-dimension float32 vector for
a rectified card view, suitable for storage in the
`card_instances.reid_embedding` BLOB column.

The current backbone is DINOv2 ViT-S/14 (384-dim). Choice of backbone
is internal to this module; callers should not assume a specific
dimension.

Closes V4_CONCERNS §1.6.
"""
from __future__ import annotations

import numpy as np

# Import the existing DINOv2 embedder from deduplicator. The exact
# import path depends on how Wave 2 wired it — adjust to match.
from card_capture.deduplicator import _dinov2_embed  # noqa: F401  (placeholder)


def compute_reid_embedding(rectified_bgr: np.ndarray) -> np.ndarray:
    """Compute a ReID embedding for a rectified card view.

    Args:
        rectified_bgr: HxWx3 uint8 BGR image (typically 1050x750).

    Returns:
        A 1D float32 numpy array (currently 384-dim DINOv2 output).
    """
    return _dinov2_embed(rectified_bgr).astype(np.float32)
```

**Important:** the `from card_capture.deduplicator import _dinov2_embed`
above is a stand-in. Step 2.2's investigation determines the actual
name. If DINOv2 is inline inside a class method, refactor it out into
`card_capture.deduplicator._dinov2_embed` (or similar) as part of this
step so the helper has something to call.

- [ ] **Step 2.6: Hook the embedding into the storage path**

Edit `pipeline/steps/store.py`. Find where `card_instances` rows are
inserted. Before the INSERT, compute the embedding for the canonical
fused view and pass it as a parameter:

```python
from card_capture.ml.embeddings import compute_reid_embedding

# When inserting a card_instance:
embedding = compute_reid_embedding(fused_canonical_bgr)
cursor.execute(
    "INSERT INTO card_instances (..., reid_embedding) VALUES (..., ?)",
    (..., embedding.tobytes()),
)
```

The exact SQL and surrounding code depend on the post-A1 shape of
`store.py`. Read it carefully; the change should be additive.

- [ ] **Step 2.7: Document the policy**

At the top of `src/card_capture/deduplicator.py`, add a docstring:

```python
"""Cross-card deduplication.

ReID embedding policy (V4_CONCERNS §1.6 / §2.16):
- Every `card_instances` row carries a `reid_embedding` regardless of
  tracker backend. The embedding is computed by
  `card_capture.ml.embeddings.compute_reid_embedding` on the rectified
  canonical view at storage time (see `pipeline/steps/store.py`).
- ByteTrack does not produce a ReID embedding on its own; the storage-
  time helper is the canonical source.
- BoT-SORT's tracker-level embedding is currently ignored; if BoT-SORT
  is revived as the default, reconcile this in Wave 5.
"""
```

- [ ] **Step 2.8: Run the tests — expect PASS**

```bash
pytest tests/ml/test_embeddings.py -v
pytest tests/pipeline/test_reid_embedding_populated.py -v
python -m pytest tests/ -q
```

Expected: `test_embeddings.py` green. Integration test green if fixture
present; skipped otherwise (acceptable for CI).

- [ ] **Step 2.9: Commit and open PR**

```bash
git add src/card_capture/ml/embeddings.py \
        src/card_capture/deduplicator.py \
        pipeline/steps/store.py \
        tests/ml/test_embeddings.py \
        tests/pipeline/test_reid_embedding_populated.py
git commit -m "feat(ml): populate reid_embedding for every card_instances row

Extracts the DINOv2 embedding call from deduplicator into a reusable
helper; pipeline/steps/store.py now computes the embedding on the
rectified canonical view at storage time. Result: the column is
populated regardless of tracker backend (ByteTrack default would
otherwise leave it NULL — see §1.6 for the policy reasoning).

Closes V4_CONCERNS §1.6.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
gh pr create --title "[Wave 4 — Surface C] ReID embedding populated for every card (C2)" --body "$(cat <<'EOF'
## Summary
- Extracts DINOv2 embedding call into card_capture.ml.embeddings.
- pipeline/steps/store.py populates reid_embedding for every card_instance regardless of tracker.
- Integration test asserts no NULL values for both ByteTrack and BoT-SORT runs.

Closes V4_CONCERNS §1.6.
Coordinates with A1 (rebase already done; only the targeted hook into store.py).

## Test plan
- [x] new tests added: test_embeddings.py, test_reid_embedding_populated.py
- [x] pytest tests/ green locally
- [x] CI green
EOF
)"
```

Wait for merge.

---

## Task 3: Update V4_CONCERNS.md and final verification

- [ ] **Step 3.1: Move §1.5, §1.6 to §2**

Edit `V4_CONCERNS.md`:

- §1.5 body → `**Resolved (see §2.16)**`. Add §2.16 with PR number for C1.
- §1.6 body → `**Resolved (see §2.17)**`. Add §2.17 with PR number for C2.

```bash
git add V4_CONCERNS.md
git commit -m "docs(wave4): mark §1.5, §1.6 resolved by Surface C

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3.2: Report completion**

Surface C is done.

---

## Self-Review Checklist

- [ ] C1, C2 merged.
- [ ] `V4_CONCERNS.md` §1.5, §1.6 moved to §2.
- [ ] CI green on `main`.
- [ ] `FBPredictor` raises without checkpoint; `is_available` works both ways.
- [ ] `reid_embedding` integration test passes for both tracker backends (or skips cleanly if fixture absent).
