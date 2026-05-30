# V5.5 Back-Half Baseline

Date: 2026-05-29
Tag (proposed): `v55-back-half-complete`
Video: `tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV`
Detector: `docaligner`
Git SHA: `346db8690dcbff41646b3ac5a40725666d4dc081` (HEAD of `feat/v55-back-half-wiring` at template creation)

> **Status: TEMPLATE / PLACEHOLDER.**
> This document is the merge-gating evidence required by §15 of the back-half
> plan. It was scaffolded in a session without access to the golden-corpus
> `IMG_5872.MOV` fixture and without CUDA/MPS hardware capable of running the
> real `docaligner` detector at production FPS. The schema, gates, and
> comparison columns are fixed below; numeric cells marked `TODO` MUST be
> filled in by an operator who can execute the commands in
> "Reproduction" against the real fixture before this branch is merged.

---

## Reproduction

### Step 1 — Process the golden video

```bash
.venv/bin/python -m card_capture.cli process \
    tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV \
    --output-dir card_capture_output/back-half-baseline \
    --db card_capture_output/back-half-baseline/cards.sqlite \
    --detector docaligner
```

Expected: process completes successfully; `cards.sqlite` contains
`card_instances` rows.

### Step 2 — Run the harness

```bash
.venv/bin/python -m card_capture.cli harness run \
    --baseline v1 \
    --db card_capture_output/back-half-baseline/cards.sqlite \
    --truth-dir tests/fixtures/golden_corpus/IMG_5872/
```

Record all 5 metrics emitted: `card_recall`, `card_precision`,
`side_accuracy`, `image_quality (SSIM)`, `image_quality (PSNR)`.

### Step 3 — Fill in the metrics table below and re-commit

Replace each `TODO` cell with the harness value, compute Δ vs. the V4
baseline, and mark Pass/Fail per the per-metric gate.

---

## Metrics

V4 baseline values are sourced from
[`docs/superpowers/plans/v5-5/baseline-results.md`](./baseline-results.md)
(structural `fake`-detector run on `IMG_5872.MOV`, Git SHA `964852b5`).

| Metric           | V4 baseline | V5.5 back-half | Δ      | Gate (±) | Pass? |
|------------------|-------------|----------------|--------|----------|-------|
| card_recall      | 0.1667      | TODO           | TODO   | 0.05     | TODO  |
| card_precision   | 1.0000      | TODO           | TODO   | 0.05     | TODO  |
| side_accuracy    | 1.0000      | TODO           | TODO   | 0.05     | TODO  |
| SSIM             | 0.4964      | TODO           | TODO   | 0.05     | TODO  |
| PSNR             | 8.0904      | TODO           | TODO   | 0.5      | TODO  |

> **Caveat on V4 column.** The V4 baseline above was captured with the
> structural `fake` detector at 1 FPS, not the production `docaligner`
> detector. If the deltas against that column are large (especially for
> `card_recall`, SSIM, and PSNR, which are detector-sensitive), re-establish
> the V4 baseline with `--detector docaligner` and update both columns before
> declaring a gate failure.

---

## Notes

- Run on: `TODO: <machine model / GPU / OS>`.
- Wallclock total: `TODO: <seconds>`.
- `crops/` count: `TODO: <n>`.
- Active tracks peak: `TODO: <n>`.
- Resident memory peak: `TODO: <MB>`.
- Notable behaviors observed (specific cards missed, front-back swaps,
  fusion artefacts, etc.): `TODO`.

---

## Gate verdict

`TODO: PASS / FAIL`.

- If PASS — tag `v55-back-half-complete` at the recorded SHA and proceed
  to the §15 sign-off.
- If FAIL — link the follow-up issue(s) here and either rework the
  failing stage or document the variance as accepted (with rationale)
  before merge.

---

## Schema reference

The columns above are fixed by Phase 14 of
`docs/superpowers/plans/2026-05-29-v55-back-half-plan.md`. Do not add or
remove rows in the Metrics table without amending that plan; downstream
trend tracking depends on a stable schema.
