# Dynamic Novelty Gate Design

**Date:** 2026-05-17
**Goal:** Automatically enable or disable the background-novelty track-pruning gate based on whether the per-video novelty score distribution indicates the background model is discriminating.

---

## Problem

The score step (`pipeline/steps/score.py`) has a pruning gate that compares each track's median novelty score against a threshold. In v4.1 the threshold is hardcoded to `< 0.0` — impossible, so the gate never fires. The result: acrylic-stand false positives survive all the way to output on stand-style videos where the background model could have caught them.

The gate was disabled because it breaks hand-held / pack-opening videos where the background is never empty — all novelty scores are high, and a static threshold would prune real cards.

---

## Solution

Analyse the distribution of `ctx.observed_novelty_scores` (already collected during the novelty step) at the start of the score step. Activate the gate only when the distribution shows the background model is actually discriminating. Use a fixed novelty floor when active; leave the threshold at `-1.0` (never prunes) when not.

---

## Detection Criterion

```python
def _novelty_gate_useful(scores: list[float]) -> bool:
    if len(scores) < 5:
        return False
    import numpy as np
    arr = np.array(scores, dtype=np.float32)
    return float(arr.std()) > 0.15 and float(arr.min()) < 0.35
```

All three conditions must hold:

| Condition | Why |
|---|---|
| `len >= 5` | Too few detections to judge distribution |
| `std > 0.15` | Background model is discriminating — not all scores clustered near 1.0 |
| `min < 0.35` | At least some detections look like background — something to filter |

**Stand-style video:** background model captures empty stand. Stand detections score `0.05–0.20`, real card detections `0.60–0.95`. `std ≈ 0.25`, `min ≈ 0.08` → gate on → stand tracks pruned.

**Hand-held video:** everything is novel vs background. All scores cluster `0.70–0.95`. `std ≈ 0.05`, `min ≈ 0.65` → gate off → nothing pruned.

---

## Score Step Change

Replace the hardcoded line in `pipeline/steps/score.py`:

```python
# Before
should_prune = (bg_model is not None) and (median_novelty < 0.0)

# After
gate_useful = _novelty_gate_useful(ctx.observed_novelty_scores)
threshold = ctx.novelty_floor if gate_useful else -1.0
should_prune = (bg_model is not None) and gate_useful and (median_novelty < threshold)
```

Add `gate_useful` and `threshold` to the print line for observability:

```python
print(
    f"[Stage: Score] | {len(scored_tracks)} tracks scored "
    f"| {len(pruned_instance_ids)} pruned | {active_count} active"
    f" | novelty_gate={'on' if gate_useful else 'off'} threshold={threshold:.2f}"
)
```

---

## New Config Field

**`src/card_capture/config.py`** and **`pipeline/steps/start.py`**:

```python
novelty_floor: float = 0.30
```

The prune threshold used when the gate is active. Tracks with `median_novelty < 0.30` are pruned. Stand detections typically score `0.05–0.20`; real cards `0.60–0.95`. A floor of `0.30` gives a comfortable margin in both directions.

---

## Files Changed

| File | Change |
|---|---|
| `src/card_capture/config.py` | Add `novelty_floor: float = 0.30` |
| `pipeline/steps/start.py` | Add `novelty_floor: float = 0.30` to `RunContext`; wire `novelty_floor=cfg.novelty_floor` in `init_run()` |
| `pipeline/steps/score.py` | Add `_novelty_gate_useful()` helper; replace `< 0.0` threshold with dynamic gate |

No changes to `novelty.py`, `track.py`, `refine.py`, or any intermediate dataclass. `ctx.observed_novelty_scores` already flows through Metaflow serialization.

---

## Testing

One unit test for `_novelty_gate_useful` in `tests/pipeline/test_score_novelty_gate.py`, covering four cases:

| Case | scores | Expected |
|---|---|---|
| Bimodal (stand + cards) | `[0.05, 0.08, 0.85, 0.90, 0.92]` | `True` |
| All high (hand-held) | `[0.82, 0.88, 0.91, 0.87, 0.85]` | `False` (std too low) |
| Too few samples | `[0.05, 0.90]` | `False` |
| High std but min not low enough | `[0.40, 0.90, 0.91, 0.40, 0.88]` | `False` (min not < 0.35) |

No new fixtures, no video files, no mocking.

---

## Observability

After the change, every run logs whether the gate fired and at what threshold:

```
[Stage: Score] | 12 tracks scored | 2 pruned | 10 active | novelty_gate=on threshold=0.30
[Stage: Score] | 8 tracks scored | 0 pruned | 8 active | novelty_gate=off threshold=-1.00
```
