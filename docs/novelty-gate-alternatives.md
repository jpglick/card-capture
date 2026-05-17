# Novelty Gate: Alternatives and Streaming Considerations

The background novelty gate (Stage 4) exists as a workaround for YOLO false positives
on empty card stands.  It compares each frame against a background model and drops
frames that look too similar to an empty workspace.  For streaming / pack-opening
footage this assumption breaks down completely — the background is never empty.

---

## What the gate actually does

- Builds a pixel-mean background model from sampled "empty" frames
- Computes L2 distance between each incoming frame and the background model
- Drops frames below a novelty threshold (`background_novelty_threshold = 0.08`)
- Goal: prevent phantom tracks of the empty stand from becoming exported cards

The gate fires **per frame**, before tracking, so a false positive here silently
discards frames that may contain real cards.

---

## Why it fails for streaming footage

1. **The background is never empty.** Card mats, packs, wrappers, and previously
   revealed cards accumulate throughout the video.  The "background model" describes
   a busy scene, not an empty stand.

2. **Previously revealed cards in frame poison the model.** Cards left on a hit
   display while opening continues look card-like but are part of the background.
   The novelty gate may then suppress a newly placed card because it resembles
   the existing background content.

3. **Complex backgrounds are less likely to fool YOLO anyway.** The specific failure
   mode the gate was built for — an empty stand with rectangular geometry similar to
   a card — is much less likely with streaming backgrounds.

---

## Alternative approaches

### 1. Trust YOLO's confidence threshold (simplest)

An empty stand or complex background should not trigger YOLO-OBB at ≥0.5 confidence
unless it was in the training data.  If it does, the fix belongs in YOLO's training
data (hard-negative mining with empty-stand and background frames), not in a
downstream pixel-distance gate.  This is the root cause; the novelty gate is a
symptom treatment.

**Tradeoff:** Requires clean YOLO training data.  Currently the empty stand does
trigger detections in stand-style videos, so removing the gate without fixing YOLO
would re-introduce phantom tracks there.

---

### 2. Track-quality floor (recommended complement)

Apply a minimum quality score to finished tracks after fusion rather than gating
per frame before tracking.  A phantom stand track would score low on:

- **Complexity** — stand surface is flat, low texture
- **Border purity** — no clean card border present
- **YOLO confidence** — the detections that seeded the track were marginal

Example: drop any track where `total_score < 0.30`.

**Advantage over novelty gate:** Evaluates the fused, high-quality result rather
than noisy individual frames.  Less likely to suppress real cards.  Works for all
video styles.

---

### 3. Minimum detection confidence per track

Require that at least N frames in a track exceed a stricter YOLO confidence floor
(e.g., 0.65).  Phantom stand tracks tend to cluster near the detection threshold
(0.50–0.55).  Real card tracks have detections consistently in the 0.70–0.95 range
— visible in telemetry as `candidate_confidence_p50`.

Can be implemented as a post-tracking prune step with no architecture changes.

---

### 4. Valley-split aware frame exclusion

For pack openings the sampler already identifies valley split boundaries (card swap
moments).  Frames in an inter-split interval where YOLO fires zero detections can
be classified as "empty interval" and skipped without a pixel-distance comparison.

The novelty gate is only meaningful when YOLO fired but the frame might still be
empty — a narrow case.  If YOLO didn't fire there is nothing to gate.  If YOLO
fired at ≥0.5 the gate is second-guessing a semantic model with a pixel heuristic.

---

### 5. Appearance-based post-track gate (longer term)

After Stage 9 produces a fused canonical image, run a lightweight binary classifier
(MobileNet-style, similar to the existing presence classifier) to confirm the result
looks like a card.  Advantages:

- Operates on a high-quality fused image, not a noisy scan frame
- Can be trained on actual false-positive examples the pipeline has produced
- Does not require knowing what the background looks like
- Handles all video styles uniformly

Cost: classification runs on every track output rather than gating frames early.
For typical volumes (5–20 tracks per video) this is negligible.

---

## Recommended approach by video style

| Replacement technique | Streaming / pack opening | Stand-style recording |
|---|---|---|
| YOLO confidence threshold alone | Sufficient | Insufficient (stand can fool it) |
| Track quality floor | Good complement | Good complement |
| Minimum track confidence | Good complement | Good complement |
| Valley-split aware exclusion | Natural fit | Already implicit |
| Appearance post-track gate | Best long-term | Best long-term |
| Novelty gate (current) | **Disable** | Keep as option |

---

## Immediate actionable change

Add `novelty_gate` as an explicit config flag (default `true` for backward
compatibility) and skip Stage 4 entirely when false.

```json
{ "novelty_gate": false }
```

In `pipeline/steps/novelty.py`, guard the gate logic:

```python
if ctx.novelty_gate:
    candidates = _apply_novelty_gate(candidates, ctx)
```

This unblocks streaming videos today without touching any algorithm and makes
the gate's presence explicit rather than always-on.

**Longer term:** Fix YOLO's false-positive rate on empty stands via hard-negative
training, add a track-quality floor, and deprecate the novelty gate entirely in
favour of the appearance post-track gate.

---

## Related files

| File | Relevance |
|---|---|
| `src/card_capture/presence/background_novelty.py` | Novelty gate implementation |
| `pipeline/steps/novelty.py` | Gate integration in pipeline |
| `src/card_capture/scoring.py` | Quality scorer (track quality floor lives here) |
| `src/card_capture/config.py` | Add `novelty_gate: bool = True` |
| `docs/presence-classifier-training.md` | Retraining the presence classifier |
