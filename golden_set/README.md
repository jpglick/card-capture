# Golden Set

Contains ground-truth labels and reference frames for regression testing.

## Structure

```
golden_set/
  videos/                  Per-video truth files
    <video_id>.truth.json  Canonical truth file location
  README.md                This file
```

## Reference Frames

`reference_frames/` subdirectories contain machine-generated frame stills
used for `image_quality` metric comparisons. They are **not committed**;
regenerate them with:

```bash
python scripts/generate_reference_frames.py --golden-set golden_set/
```

## Truth File Format

See `docs/contracts/truth-schema.md` for the full schema specification.

## Coverage Targets

- 15 total videos by Wave 1 gate.
- Must include: clean run, glare, foil, hand occlusion, fast swaps,
  edge-on flips, dark workspace, bright workspace, mixed orientations,
  partial visibility, multi-card-in-frame.

## Current Videos

- **IMG_5872**: Clean run with 6 cards.
