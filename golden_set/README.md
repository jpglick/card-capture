# Golden Set

Labeled videos used by the regression harness. Each subdirectory under
`videos/` is one labeled video:

```
videos/<video_id>/
    truth.json           # Contract 4 schema
    source.mov.symlink   # symlink or note pointing to source video
    reference_frames/    # hand-picked reference frames for SSIM
```

`_index.txt` lists all labeled video ids, one per line. The harness CLI
reads this file when `--videos` is omitted.

Coverage targets:
- 15 total videos by Wave 1 gate.
- Must include: clean run, glare, foil, hand occlusion, fast swaps,
  edge-on flips, dark workspace, bright workspace, mixed orientations,
  partial visibility, multi-card-in-frame.

## Current Videos
- **IMG_5872**: Clean run with 6 cards.
