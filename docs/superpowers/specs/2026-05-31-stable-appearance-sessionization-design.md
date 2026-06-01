# Stable Appearance Sessionization Design

## Context

`IMG_5922` contains 26 card-front presentations. It contains no card backs.
Some fronts are visually identical because separate physical cards can be
duplicates. The pipeline must preserve those separate physical instances.

The previous reset analysis incorrectly assumed that approximately 46
appearance changes represented front/back pairs. The enriched diagnostic in
`scripts/diag_swap_signals.py` disproved that assumption:

| Measurement | Count |
| --- | ---: |
| Detected frames | 378 |
| Sampled frames | 437 |
| Raw DINOv2 jumps above `0.30` | 56 |
| Stable plateaus with at least 3 detections | 44 |
| Repeated holder/null plateaus | 18 |
| Remaining card-front plateaus | 26 |
| Ground-truth card fronts | 26 |

The reset problem is therefore presentation sessionization, not front/back
pairing and not downstream deduplication.

## Goals

- Emit one presentation session for each stable card-front plateau.
- Preserve separate physical cards even when their fronts are visually
  identical.
- Handle direct in-place front-to-front replacement.
- Handle optional holder/null plateaus without emitting them as cards.
- Reuse existing DINOv2 embeddings without adding another model.
- Keep PyTorch and Kornia work inside the guarded runtime worker boundary.

## Non-Goals

- Do not infer physical identity from visual similarity alone.
- Do not require a detection gap, centroid jump, or holder flash.
- Do not use front/back resolution to repair tracker over-splitting.
- Do not remove visual dedup metadata used for cataloging or cross-video links.

## Rejected Approaches

### Detection Gaps

The detector bridges most swaps. Only 9 transitions have gaps of at least 2
sampled frames and only 3 have gaps of at least 3 frames.

### Centroid Jumps

Cards are replaced in place. Centroid motion is not a reliable session
boundary.

### Raw Appearance Jumps

A raw DINOv2 distance threshold above `0.30` fires 56 times. Holder exposure,
hand occlusion, and transition crops cause multiple jumps around one card
replacement.

### Required Holder State

Holder/null visibility is mixed. Some card replacements are direct. A holder
appearance is useful evidence when present but cannot be required.

## Design

### Two-Pass Sessionizer

Run sessionization over the complete sampled sequence before assigning final
`session_id` values. Tracking already receives the complete frame sequence, so
offline classification within the in-process stage is acceptable and avoids
making irreversible decisions before a candidate plateau stabilizes.

### Pass 1: Stable Plateau Formation

Build plateaus from the top-confidence card detection for each sampled frame.
Each candidate uses an existing normalized DINOv2 embedding.

Maintain an active plateau and a buffered pending plateau:

1. Continue the active plateau while embedding distance to its representative
   is at most `0.15`.
2. Buffer a possible replacement when distance from the active representative
   exceeds `0.30`.
3. Confirm a pending plateau after at least 3 mutually similar detections, each
   within `0.15` of its pending representative.
4. Treat distances in the ambiguous `(0.15, 0.30]` range as transition noise.
   They may be retained as candidate frames for quality selection but must not
   create a session boundary.
5. Allow a confirmed plateau to follow another confirmed plateau directly.
   A detection gap or null plateau is not required.

The representative is the normalized centroid of accepted embeddings in the
plateau. Buffering preserves early frames until the plateau is confirmed.

### Pass 2: Conservative Bridge Suppression

Cluster confirmed plateau representatives using DINOv2 distance at most
`0.15`. Classify a recurring cluster as a holder/null bridge only when all of
the following hold:

- It occurs at least 3 times in non-adjacent positions.
- At least 80% of its occurrences sit between other stable plateaus.
- For at least 80% of those interior occurrences, the neighboring plateaus
  belong to different presentation clusters.
- Its per-video novelty or duration distribution supports a bridge role:
  - Median novelty exceeds the median novelty of neighboring retained plateaus
    by at least `0.05`; or
  - Median plateau length is at most 75% of the median length of neighboring
    retained plateaus.

Novelty and duration are supporting evidence, not standalone gates. This is
required because visually identical physical cards are valid and because
holder visibility is mixed.

Suppression is conservative: if bridge classification is not confident, retain
the plateau as a physical card presentation. False retention is preferable to
silently dropping a physical card.

On `IMG_5922`, this pass suppresses the repeated holder/null cluster with 18
occurrences and retains 26 card-front plateaus.

### Session Assignment

Assign a new monotonically increasing `session_id` to every retained stable
plateau. Do not merge retained plateaus when their representative embeddings
match. Matching retained plateaus can represent separate physical duplicate
cards.

Reset BoT-SORT at retained plateau boundaries. Suppressed bridge detections
must not produce card tracks. Existing centroid and gap signals may remain as
telemetry or supporting evidence but must not be authoritative reset triggers.

### Embedding Boundary

The sessionizer consumes normalized NumPy embeddings. It must not invoke
PyTorch from the producer or main thread. Implementation must reuse embeddings
produced inside the guarded worker context or extend the worker output to
include them.

### Resolve and Dedup Semantics

Front/back resolution remains downstream and must not be used to explain or
collapse sessionization errors. For a front-only video, each retained session
is a physical card-front presentation.

Visual dedup remains metadata-only:

- Intra-run visual matches may record that two physical instances show the
  same design.
- Cross-video visual matches may record catalog relationships.
- Neither case may remove a retained physical instance from final output.

The current dedup stage stores all `fused_canonicals`, which preserves output
instances. Tests must lock this behavior down.

## Configuration

Introduce explicit configuration values rather than overloading centroid
settings:

| Setting | Initial value | Purpose |
| --- | ---: | --- |
| `appearance_same_threshold` | `0.15` | Extend a stable plateau |
| `appearance_change_threshold` | `0.30` | Start buffering a possible replacement |
| `appearance_confirm_frames` | `3` | Confirm a stable plateau |
| `bridge_min_occurrences` | `3` | Minimum recurrence for conservative null suppression |
| `bridge_position_ratio` | `0.80` | Required fraction of interior bridge occurrences |
| `bridge_neighbor_change_ratio` | `0.80` | Required fraction bracketed by distinct appearances |
| `bridge_novelty_margin` | `0.05` | Per-video novelty support margin |
| `bridge_max_length_ratio` | `0.75` | Per-video duration support ratio |

Per-video novelty and duration support should be derived from observed plateau
distributions. Do not hard-code the `IMG_5922` holder novelty median as a
universal threshold.

## Telemetry

Emit enough data to audit sessionization:

- Raw appearance-jump count.
- Confirmed plateau count.
- Suppressed bridge plateau count.
- Retained presentation count.
- Plateau frame range, length, representative cluster, median novelty, and
  suppression reason.
- Final `session_id` boundary frame indices.

## Verification

Add deterministic unit tests for:

- Direct front-to-front replacement creates two sessions after confirmation.
- A repeated holder plateau between cards is suppressed.
- Mixed direct and holder-mediated replacements produce the expected sessions.
- An isolated unusual card plateau is retained.
- Repeated visually identical card plateaus remain separate physical sessions.
- Ambiguous transition frames do not create sessions.
- Dedup metadata does not reduce final physical-instance output count.
- No new embedding inference occurs outside the guarded worker boundary.

Run:

```bash
pytest -m "not quarantine"
```

Run the enriched `IMG_5922` diagnostic and the full pipeline manually in a
local terminal for final GPU behavior and real-world timing verification.
