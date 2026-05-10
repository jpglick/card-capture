# Setup / Onboarding Page Design

**Date:** 2026-05-10
**Scope:** Single route + template addition to the existing Review UI.

## Goal

Add a `/setup` page to the Review UI that explains the four manual bootstrap steps (process a video, label corpus, run harness, capture baseline) and shows live detected progress on each step.

## Route

`GET /setup` added to `create_app()` in `src/card_capture/review.py`.

**Status computation (server-side, at request time):**

| Variable | Source | Meaning |
|---|---|---|
| `video_count` | `SELECT COUNT(*) FROM videos` | Videos processed by the pipeline |
| `truth_count` | glob `tests/fixtures/golden_corpus/*/*.truth.json` | Labeled videos in corpus |
| `any_report` | glob `reports/*.json` | Harness run at least once |
| `has_baseline` | `reports/baseline_v3.json` exists | Baseline captured |

## Template

`src/card_capture/templates/setup.html` — four checklist items rendered with `{% if %}` badge logic:

1. **Process at least one video** — badge green when `video_count > 0`, shows count. Command: `card-capture process <video.mp4>`.
2. **Label corpus videos** — badge green when `truth_count > 0`, shows count of labeled vs total processed. Instructions: open `/label/<id>` in the review UI for each video.
3. **Run harness once** — badge green when `any_report`. Command: `card-capture harness run` or `make harness`.
4. **Capture baseline** — badge green when `has_baseline`. Command: `mv reports/<sha>.json reports/baseline_v3.json` or `make baseline`.

## Navigation

"Setup" link added to the `<nav>` / header section of all three existing templates: `review.html`, `timeline.html`, `labeling.html`.

## Files changed

- **Modify:** `src/card_capture/review.py` — add `GET /setup` route (~15 lines)
- **Create:** `src/card_capture/templates/setup.html`
- **Modify:** `src/card_capture/templates/review.html` — add nav link
- **Modify:** `src/card_capture/templates/timeline.html` — add nav link
- **Modify:** `src/card_capture/templates/labeling.html` — add nav link

## Non-goals

- No polling / auto-refresh.
- No "run harness" button in the UI (command-line only for now).
- No video list with per-video labeling links.
