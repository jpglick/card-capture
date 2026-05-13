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
    """Group duplicate card instances.

    Args:
        ctx:              RunContext from the start step.
        fused_canonicals: List of fused image metadata from the fuse step.

    Returns:
        ``DedupOutput`` with grouped duplicates.
    """
    from card_capture.deduplicator import VisualDeduplicator
    from card_capture.identity.embedding_distance import embedding_same_card_score

    deduplicator = VisualDeduplicator()
    
    # Intra-run deduplication
    # (Matches cards across different sessions in the same video)
    
    dedup_groups: List[Dict[str, Any]] = []
    processed_ids: set = set()
    
    # We'll use a simple greedy clustering for now
    for i, f1 in enumerate(fused_canonicals):
        id1 = f1["instance_id"]
        if id1 in processed_ids:
            continue
            
        group = {
            "canonical_instance_id": id1,
            "duplicate_instance_ids": [],
            "hamming_distances": {}
        }
        processed_ids.add(id1)
        
        for j, f2 in enumerate(fused_canonicals[i+1:]):
            id2 = f2["instance_id"]
            if id2 in processed_ids:
                continue
            
            same = False
            
            # 1. Try embedding
            emb1 = f1.get("appearance_vector")
            emb2 = f2.get("appearance_vector")
            if emb1 and emb2:
                same = embedding_same_card_score(np.array(emb1), np.array(emb2), threshold=0.5)
            
            # 2. Fallback to pHash
            if not same:
                h1 = f1["primary_hash"]
                h2 = f2["primary_hash"]
                ham = deduplicator.hamming_distance(h1, h2)
                if ham <= 30: # Looser threshold for cross-session
                    same = True
                    group["hamming_distances"][id2] = float(ham)
            
            if same:
                group["duplicate_instance_ids"].append(id2)
                processed_ids.add(id2)
                
        dedup_groups.append(group)
        
    return DedupOutput(
        dedup_groups=dedup_groups,
        dedup_distances={} # Flat mapping if needed
    )
