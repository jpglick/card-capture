"""Threshold calibration for DINOv2 deduplication.

Sweeps cosine similarity thresholds against labeled ``dedup_clusters``
to find the optimal Adjusted Rand Index (ARI).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from card_capture.data.connection import read_connection
from card_capture.data.sql_queries import DEDUP_CONFIRMED_CLUSTERS, DEDUP_INSTANCE_FUSED_BY_TRACK
from card_capture.ml.inference.dino_dedup import DinoDeduplicator


def calibrate(*, db_path: Path, variant: str = "vits14") -> dict:
    """Find the threshold that maximizes ARI on confirmed clusters."""
    # 1. Load confirmed clusters from DB
    with read_connection(db_path) as conn:
        rows = conn.execute(DEDUP_CONFIRMED_CLUSTERS).fetchall()

    if not rows:
        return {"error": "No confirmed clusters found for calibration."}

    # 2. Collect all images and their GT cluster assignments
    gt_map = {} # instance_id -> cluster_id
    instance_to_path = {}
    
    with read_connection(db_path) as conn:
        for r in rows:
            cid = r["cluster_id"]
            members = json.loads(r["confirmed_member_ids_json"])
            for mid in members:
                gt_map[mid] = cid
                # Find image path for this instance
                card_row = conn.execute(DEDUP_INSTANCE_FUSED_BY_TRACK, (mid,)).fetchone()
                if card_row and Path(card_row["fused_image_path"]).exists():
                    instance_to_path[mid] = Path(card_row["fused_image_path"])

    if not instance_to_path:
        return {"error": "No valid image paths found for confirmed instances."}

    mids = list(instance_to_path.keys())
    paths = [instance_to_path[m] for m in mids]

    # 3. Sweep thresholds
    deduper = DinoDeduplicator(variant=variant)
    deduper.add_instances(mids, paths)

    from sklearn.metrics import adjusted_rand_score

    best_ari = -1.0
    best_t = 0.0
    
    thresholds = np.linspace(0.80, 0.99, 20)
    results = []

    # Get GT labels in order of mids
    y_true = [gt_map[m] for m in mids]

    for t in thresholds:
        deduper.threshold = float(t)
        clusters = deduper.cluster_all()
        
        # Convert clusters to label array
        y_pred = np.zeros(len(mids), dtype=int)
        for cid, cluster in enumerate(clusters):
            for mid in cluster:
                # find index of mid in mids
                idx = mids.index(mid)
                y_pred[idx] = cid
                
        ari = adjusted_rand_score(y_true, y_pred)
        results.append({"threshold": float(t), "ari": float(ari)})
        
        if ari > best_ari:
            best_ari = ari
            best_t = t

    print(f"Optimal DINOv2 threshold: {best_t:.3f} (ARI: {best_ari:.4f})")
    
    return {
        "best_threshold": float(best_t),
        "best_ari": float(best_ari),
        "sweep": results
    }
