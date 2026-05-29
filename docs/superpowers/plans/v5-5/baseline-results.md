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

## V5.5 Completion Baseline — 2026-05-28

- **Tag:** `v55-complete`
- **Default lane:** `pytest tests/ -m 'not quarantine' -q` → **591 passed, 29 skipped, 12 deselected in 30.90s**
- **Architecture lane:** `pytest tests/architecture/ -v` → **5 passed** (`test_gpu_strict_calls`, `test_import_linter`, `test_metaflow_absent`, `test_raw_sql_outside_data`, `test_skip_audit`)
- **Perf smoke:** `pytest tests/performance/test_perf_harness_smoke.py -v` → **PASS**
- **Import Linter:** `PYTHONPATH=src:. lint-imports` → **5 kept, 0 broken** (227 files, 865 dependencies analyzed)
- **Raw-sqlite3 outside data:** 0 callsites
- **Metaflow imports outside vendored env:** 0 callsites
- **Vast.ai imports anywhere:** 0 callsites

All gaps identified in the 2026-05-28 verification of the original V5.5 plan are closed.

### Plan deviations recorded at completion

The closeout verification surfaced two textual discrepancies between the parent completion plan and the implemented `.importlinter`. Both were resolved in favor of the implementation; neither weakens enforcement.

1. **Contract count: 5, not 6.** The parent plan's gauntlet expects "6/6 contracts kept", but the refactoring spec only ever defined five contract sections: `no-sqlite3-outside-data`, `no-provider-sdk-outside-platforms`, `strict-gpu-no-image-io`, `layered`, and `no-metaflow`. These five cover every distinct architectural concern (data-layer boundary, platform-adapter boundary, GPU-strict file IO, layer order, dead-framework removal). The "6/6" reference in `2026-05-28-v5-5-completion.md` is a documentation typo from an earlier draft and does not correspond to a missing contract.
2. **Strict-GPU forbidden module: `cv2`, not `cv2.imgcodecs`.** The refactoring plan's example config listed `PIL`, `PIL.Image`, and `cv2.imgcodecs` as the strict-GPU forbidden modules. The implemented config uses `PIL` and `cv2`. This is intentional: in CPython, `cv2` is a flat extension module — `cv2.imgcodecs` is not an importable submodule, so an import-linter entry for it would be a no-op. Banning all of `cv2` from `card_capture.runtime.strict_gpu` is strictly more conservative than the spec, and the AST-based scanner at `tests/architecture/test_gpu_strict_calls.py` continues to enforce `cv2.imread`/`cv2.imwrite`/`cv2.VideoCapture` call-site bans separately with its own allow-list. Banning `PIL` alone covers `from PIL import Image` (which is what real code uses); a separate `PIL.Image` entry would be redundant.

### Phase-by-phase commit anchors

- Phase A regression recovery: `5582a076` (rewrite `tests/test_unified_runtime.py` against `LocalPipelineRuntime`).
- Phase B static enforcement: `cc21089c` (ci-lane-commands.md), `84f9dfa1` (strict-GPU contract), and the un-gating of `tests/architecture/test_import_linter.py`.
- Phase C DAL migration: `be7e0b60` (missing repository tests), `f6a62721` (raw sqlite3 removal), `3a75e938` and `fe743363` (import-linter contract resolution), plus the `refactor(v55-phaseE): … SQL literals` series consolidating remaining string literals into `data/sql_queries.py`.
- Phase D platform adapters: `5e51a9bc` (drop RemoteRuntime), `097642c8` (`platforms.failures`), `1f1a6d88` (`platforms.manifests`), `ecf6938c` (`LocalRunner`), `e2d957b0` (Runpod/Beam runners), `c64cb729` (Vast.ai deprecation), `59be6ead` (residual vastai worker removal).
- Phase E final verification: this entry plus the always-blocking version of `tests/architecture/test_raw_sql_outside_data.py`.
