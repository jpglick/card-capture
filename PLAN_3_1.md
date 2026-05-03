# Implementation Plan: Task 3.1 (Modify `review.py`)

## Implementation
1. Add `get_fused_image(instance_id: int)` route.
2. Update `index()` to fetch and pass `fused_image_path` and `angle` for each card, joining `saved_cards` with `card_instances`.

## Verification
- Run review app and check if fused images appear and if "angle" badge exists.
