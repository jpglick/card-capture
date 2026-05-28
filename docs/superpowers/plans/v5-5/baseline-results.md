# V5.5 Local Baseline Results

Established on 2026-05-28 using a structural run (fake detector, 1 FPS) to verify harness integrity before refactoring.

**Run Metadata:**
- **Git SHA:** 964852b5
- **Video:** `IMG_5872.MOV` (Golden Set)
- **Detector:** `fake` (Structural test)

**Aggregate Metrics:**
| Metric | Value |
|---|---|
| `card_recall` | 0.1667 |
| `card_precision` | 1.0000 |
| `side_accuracy` | 1.0000 |
| `image_quality (SSIM)` | 0.4964 |
| `image_quality (PSNR)` | 8.0904 |

**Notes:**
- Baseline established using structural parameters. High-fidelity baseline requires real detector and full FPS, which must be run manually per mandates.
- All regression infrastructure (migrations, harness CLI) is verified and working.
