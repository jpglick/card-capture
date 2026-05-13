"""Step 8 — dedup: group duplicate card instances across the entire run.

Performs cross-session deduplication using primary pHash and OSNet embeddings.
Returns a list of DedupGroup items.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .start import RunContext

@dataclass
class DedupOutput:
    """Outputs of the dedup step."""
    dedup_groups: List[Dict[str, Any]]
    dedup_distances: Dict[str, float]

def run(ctx: RunContext, fused_canonicals: List[Dict[str, Any]]) -> DedupOutput:
    """Group duplicate card instances across current run and previous videos.

    Args:
        ctx:              RunContext from the start step.
        fused_canonicals: List of fused image metadata from the fuse step.

    Returns:
        ``DedupOutput`` with grouped duplicates.
    """
    import numpy as np
    from card_capture.deduplicator import VisualDeduplicator
    from card_capture.identity.embedding_distance import embedding_same_card_score
    from card_capture.storage import Storage
    from card_capture.ml.inference.dino_dedup import DinoDeduplicator

    deduplicator = VisualDeduplicator()
    storage = Storage(Path(ctx.db_path))
    
    # Thresholds for deduplication (DINOv2 cosine distance)
    # 0.0 = identical, 1.0 = orthogonal.
    # We want a tight threshold for same-card detection.
    SAME_CARD_EMB_THRESHOLD = 0.15 
    
    dedup_groups: List[Dict[str, Any]] = []
    processed_ids: set = set()
    
    # 1. Intra-run deduplication
    for i, f1 in enumerate(fused_canonicals):
        id1 = f1["instance_id"]
        if id1 in processed_ids:
            continue
            
        group = {
            "canonical_instance_id": id1,
            "duplicate_instance_ids": [],
            "hamming_distances": {},
            "embedding_distances": {},
            "cross_video_parent_id": None
        }
        processed_ids.add(id1)
        
        emb1 = f1.get("reid_embedding")
        
        # Compare to other instances in the SAME run
        for j, f2 in enumerate(fused_canonicals[i+1:]):
            id2 = f2["instance_id"]
            if id2 in processed_ids:
                continue
            
            same = False
            
            # 1. Try DINOv2 embedding (tight threshold)
            emb2 = f2.get("reid_embedding")
            if emb1 is not None and emb2 is not None:
                dist = 1.0 - np.dot(emb1, emb2) # Since they are L2-normalized
                if dist < SAME_CARD_EMB_THRESHOLD:
                    same = True
                    group["embedding_distances"][id2] = float(dist)
            
            # 2. Fallback to pHash (legacy)
            if not same:
                h1 = f1.get("primary_hash")
                h2 = f2.get("primary_hash")
                if h1 and h2:
                    ham = deduplicator.hamming_distance(h1, h2)
                    if ham <= 8: # Very tight for same-card
                        same = True
                        group["hamming_distances"][id2] = float(ham)
            
            if same:
                group["duplicate_instance_ids"].append(id2)
                processed_ids.add(id2)

        # 2. Cross-video deduplication
        # Query the database for the most similar embedding from PREVIOUS runs
        if emb1 is not None:
            # We fetch all canonical embeddings from the DB
            # TODO: Use FAISS index for large databases
            with storage._connect() as conn:
                rows = conn.execute(
                    "SELECT id, reid_embedding FROM card_instances WHERE reid_embedding IS NOT NULL AND is_duplicate_of IS NULL AND video_id != ?",
                    (ctx.video_id,)
                ).fetchall()
            
            best_db_id = None
            min_db_dist = 1.0
            
            for row in rows:
                db_id = row["id"]
                db_emb_bytes = row["reid_embedding"]
                db_emb = np.frombuffer(db_emb_bytes, dtype=np.float32)
                
                dist = 1.0 - np.dot(emb1, db_emb)
                if dist < min_db_dist:
                    min_db_dist = dist
                    best_db_id = db_id
            
            if best_db_id and min_db_dist < SAME_CARD_EMB_THRESHOLD:
                group["cross_video_parent_id"] = best_db_id
                print(f"[Stage: Dedup] | Instance {id1[:8]} matched to DB instance {best_db_id} (dist={min_db_dist:.4f})")
                
        dedup_groups.append(group)
        
    return DedupOutput(
        dedup_groups=dedup_groups,
        dedup_distances={}
    )
