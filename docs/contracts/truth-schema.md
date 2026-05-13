# Contract 4 — truth.json Schema

Per-video ground-truth file consumed by the harness, written by the labeling
UX (Surface B), validated by `harness.schema.TruthFile`.

**Status:** Frozen (Wave 1 sign-off)
**Owned by:** Surface D
**Consumed by:** Surface B (labeling UX writes truth files; regression tab reads
metrics), Surface C (training pipelines read labels)

---

## Top-level shape

```json
{
  "video_id": "practice_session_03",
  "schema_version": 1,
  "expected_cards": [ { ... } ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `video_id` | string | yes | Stable identifier for the source video. |
| `schema_version` | integer ≥ 1 | yes | Schema version; must be `1` for new files. |
| `expected_cards` | array | yes | Ground-truth card entries (may be empty). |

---

## `expected_cards[]` element

### Required fields

| Field | Type | Description |
|---|---|---|
| `card_id` | string | Stable identifier within this truth file (e.g. `"card_01"`). Must be unique within the file. |
| `front_present` | bool | The front face of this card appears in the video. |
| `back_present` | bool | The back face of this card appears in the video. |
| `physical_card_key` | string | Cross-video identity key. Cards with the same key in different videos represent the same physical card (used for dedup accuracy). |
| `is_foil` | bool | Card has holographic / foil surface. Affects fusion strategy and metric sub-grouping. |

### Optional fields

| Field | Type | Default | Description |
|---|---|---|---|
| `approx_front_window_ms` | `[int, int]` \| null | null | Inclusive `[start_ms, end_ms]` range where the front face is visible. Used for temporal matching in recall/precision. `start_ms <= end_ms` enforced by schema validator. |
| `approx_back_window_ms` | `[int, int]` \| null | null | Same for the back face. |
| `notes` | string \| null | null | Free-text annotation (e.g. `"hand occlusion at 5.2s"`). Not consumed by the harness; human-readable only. |

### Full example

```json
{
  "card_id": "card_01",
  "front_present": true,
  "back_present": true,
  "physical_card_key": "charizard_base_4_holo_1999",
  "is_foil": true,
  "approx_front_window_ms": [4200, 6100],
  "approx_back_window_ms": [6300, 7900],
  "notes": "hand occlusion at 5.2s"
}
```

---

## Complete truth.json example

```json
{
  "video_id": "practice_session_03",
  "schema_version": 1,
  "expected_cards": [
    {
      "card_id": "card_01",
      "front_present": true,
      "back_present": true,
      "physical_card_key": "charizard_base_4_holo_1999",
      "is_foil": true,
      "approx_front_window_ms": [4200, 6100],
      "approx_back_window_ms": [6300, 7900],
      "notes": "hand occlusion at 5.2s"
    },
    {
      "card_id": "card_02",
      "front_present": true,
      "back_present": false,
      "physical_card_key": "pikachu_base_58_1999",
      "is_foil": false
    }
  ]
}
```

---

## Validation

The canonical validator is `harness.schema.TruthFile` (Pydantic v2 model).

```bash
# Validate a truth file from the command line:
python -m harness.validator path/to/truth.json
```

Returns exit code `0` on success, `1` with a human-readable error on failure.

---

## Backward compatibility

`schema_version: 0` is the output of the current `templates/labeling.html`
(prior to v4). It differs from v1 in the following ways:

| Difference | Legacy (v0) | Current (v1) |
|---|---|---|
| `schema_version` field | absent | required, value `1` |
| `is_foil` field | absent | required |
| `physical_card_key` field | optional / often absent | required |
| Extra fields | `video_path`, `labeled_at` | not present |

Legacy files are read via `harness.schema.from_legacy(payload)`, which maps
the legacy shape to `TruthFile` with `schema_version=1`. Fields absent in the
legacy format are set to safe defaults (`is_foil=False`, `physical_card_key`
derived from `card_id`). The new labeling UX writes `schema_version: 1` only.

---

## File naming convention

Truth files are stored alongside the video's output directory:

```
golden_set/videos/<video_id>/<video_id>.truth.json
```

The harness also accepts a bare `truth.json` in the same directory for
backward compatibility.

---

## Contract freeze policy

Changes to this document require acknowledgment from Surface B, C, and D
owners. Minor additive changes (new optional fields) are allowed with
single-surface ack. Breaking changes (removing required fields, changing types)
require a version bump to `schema_version: 2`.
