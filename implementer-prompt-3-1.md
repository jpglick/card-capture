# Implementer Prompt: Task 3.1 (Modify `review.py`)

## Objective
Modify `src/card_capture/review.py` to support the v3 adaptive pipeline UI.

## Requirements
1. Update `index()` route in `review.py` to join `saved_cards` with `card_instances` to fetch `fused_image_path` and `angle`.
2. Add a new route `get_fused_image(instance_id: int)` that returns the fused image from `card_instances`.

## Context
- `Storage` class is used to interact with the database.
- Database schema: `saved_cards` (has `detection_id` which corresponds to `card_views.id`), `card_instances` (has `fused_image_path`, `angle`), `card_views` (links `saved_cards` to `card_instances`).
- You need to update `list_saved_cards` in `storage.py` or modify `index` in `review.py` to join. Given the instructions, modifying `review.py` to perform the join/query is requested.

## Verification
- Add a manual check in `index()` to print fetched data to see if `fused_image_path` is correctly populated.
- Run `tests/test_pipeline.py` or similar to ensure no regressions.
