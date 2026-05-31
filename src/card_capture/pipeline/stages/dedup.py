"""Stage 10: Global Dedup.

Port of V4 ``pipeline/steps/dedup.py``. Substitution: the raw
``Storage._connect()`` query becomes
``CardsRepository.find_embeddings_excluding_video(video_id=...)`` so
no raw SQL leaves ``card_capture.data``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from card_capture.pipeline.stage_metrics import emit_stage_metrics


SAME_CARD_EMB_THRESHOLD = 0.15  # DINOv2 cosine distance, identical to V4
SAME_CARD_HAMMING_MAX = 8       # pHash fallback, identical to V4


def run(state: dict, *, telemetry) -> None:
    import numpy as np
    from card_capture.deduplicator import VisualDeduplicator

    deduplicator = VisualDeduplicator()
    fused_canonicals = state.get("fused_canonicals") or []
    cards_repo = (state.get("repos") or {}).get("cards")
    current_video_id = int(state.get("video_id", 0))

    dedup_groups: List[Dict[str, Any]] = []
    processed: set = set()

    # Pre-fetch cross-video embeddings once for efficiency
    other_embeddings: List[tuple[int, bytes]] = []
    if cards_repo is not None:
        try:
            other_embeddings = cards_repo.find_embeddings_excluding_video(video_id=current_video_id)
        except Exception as exc:
            telemetry.resource_sample(
                {"event": "cross_video_embeddings_fetch_failed", "error": repr(exc)}
            )

    for i, f1 in enumerate(fused_canonicals):
        id1 = f1["instance_id"]
        if id1 in processed:
            continue

        group: Dict[str, Any] = {
            "canonical_instance_id": id1,
            "duplicate_instance_ids": [],
            "hamming_distances": {},
            "embedding_distances": {},
            "cross_video_parent_id": None,
        }
        processed.add(id1)
        emb1 = f1.get("reid_embedding")

        # Intra-run
        for f2 in fused_canonicals[i + 1:]:
            id2 = f2["instance_id"]
            if id2 in processed:
                continue

            same = False
            emb2 = f2.get("reid_embedding")
            if emb1 is not None and emb2 is not None:
                dist = 1.0 - float(np.dot(np.array(emb1), np.array(emb2)))
                if dist < SAME_CARD_EMB_THRESHOLD:
                    same = True
                    group["embedding_distances"][id2] = dist
            if not same:
                h1 = f1.get("primary_hash")
                h2 = f2.get("primary_hash")
                if h1 and h2:
                    ham = deduplicator.hamming_distance(h1, h2)
                    if ham <= SAME_CARD_HAMMING_MAX:
                        same = True
                        group["hamming_distances"][id2] = float(ham)

            if same:
                group["duplicate_instance_ids"].append(id2)
                processed.add(id2)

        # Cross-video
        if emb1 is not None and other_embeddings:
            best_id = None
            best_dist = 1.0
            emb1_arr = np.array(emb1, dtype=np.float32)
            for row_id, blob in other_embeddings:
                other = np.frombuffer(blob, dtype=np.float32)
                if other.shape != emb1_arr.shape:
                    continue
                dist = 1.0 - float(np.dot(emb1_arr, other))
                if dist < best_dist:
                    best_dist = dist
                    best_id = row_id
            if best_id is not None and best_dist < SAME_CARD_EMB_THRESHOLD:
                group["cross_video_parent_id"] = int(best_id)

        dedup_groups.append(group)

    state["dedup_groups"] = dedup_groups
    state["final_cards"] = fused_canonicals  # store stage uses this
    emit_stage_metrics(
        state,
        stage="dedup",
        metrics={"dedup_groups": len(dedup_groups), "final_cards": len(fused_canonicals)},
    )
