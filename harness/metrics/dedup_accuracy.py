"""dedup_accuracy metric.

Returns a MetricResult with breakdown keys:

- ``ari``: Adjusted Rand Index (sklearn) comparing predicted dedup clusters to
  ground-truth clusters derived from ``physical_card_key``.
- ``pair_f1``: F1 score over all pairs (whether two detections are in the same
  cluster).

Both are ``None`` when fewer than 2 matched detections exist (ARI is
undefined on trivial inputs).

Predicted clusters are read from the ``is_duplicate_of`` column in
``card_instances``: connected components of this directed tree form the
predicted clusters (roots are canonical instances).

Ground-truth clusters come from ``physical_card_key`` in ``truth.json``,
applied only to GT cards that were matched to a detection.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from harness.match import match_detections_to_truth
from harness.metrics.types import MetricResult
from harness.schema import TruthFile


@dataclass(frozen=True)
class _DedupAccuracy:
    """Internal result of the dedup_accuracy computation."""

    ari: Optional[float]
    pair_f1: Optional[float]


def dedup_accuracy(
    *, db_path: Path, truth_path: Path, video_id: str
) -> MetricResult:
    """Compute dedup accuracy (ARI + pair F1) for one video.

    Parameters
    ----------
    db_path:
        Path to ``cards.sqlite``.
    truth_path:
        Path to ``truth.json``.
    video_id:
        String video identifier.

    Returns
    -------
    MetricResult with breakdown keys ``ari`` and ``pair_f1``; both None if < 2 matched.
    """
    from sklearn.metrics import adjusted_rand_score  # type: ignore[import-untyped]

    truth = TruthFile.model_validate_json(truth_path.read_text())
    pairs = match_detections_to_truth(db_path, truth, video_id)

    # Build GT physical_card_key map for matched detections only.
    gt_key_map: dict[str, str] = {
        card.card_id: card.physical_card_key for card in truth.expected_cards
    }

    matched = [p for p in pairs if p.gt_card_id is not None and p.detection_id is not None]
    if len(matched) < 2:
        return MetricResult(breakdown={"ari": None, "pair_f1": None})

    # Ground-truth labels: physical_card_key for each matched detection.
    gt_labels = [gt_key_map[p.gt_card_id] for p in matched]  # type: ignore[index]

    # Predicted labels: cluster root for each matched detection.
    pred_cluster_map = _build_predicted_clusters(db_path)
    pred_labels = [pred_cluster_map.get(p.detection_id, p.detection_id) for p in matched]  # type: ignore[arg-type]

    ari = float(adjusted_rand_score(gt_labels, pred_labels))
    pf1 = _pair_f1(gt_labels, pred_labels)

    return MetricResult(breakdown={"ari": ari, "pair_f1": pf1})


def _build_predicted_clusters(db_path: Path) -> dict[int, int]:
    """Return a mapping from card_instance.id → cluster root id.

    Cluster root = the ancestor in the ``is_duplicate_of`` chain with
    ``is_duplicate_of IS NULL``.
    """
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, is_duplicate_of FROM card_instances"
        ).fetchall()

    # Build parent map
    parent: dict[int, Optional[int]] = {
        int(r["id"]): (int(r["is_duplicate_of"]) if r["is_duplicate_of"] is not None else None)
        for r in rows
    }

    def find_root(node_id: int, visited: set[int]) -> int:
        if node_id in visited:
            # Cycle guard — treat as its own root
            return node_id
        visited.add(node_id)
        p = parent.get(node_id)
        if p is None:
            return node_id
        return find_root(p, visited)

    return {nid: find_root(nid, set()) for nid in parent}


def _pair_f1(gt_labels: list, pred_labels: list) -> float:
    """Pair F1: F1 over all (i, j) pairs on whether they share a cluster label.

    TP: both GT and predicted say same cluster.
    FP: predicted says same, GT says different.
    FN: GT says same, predicted says different.
    """
    n = len(gt_labels)
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            gt_same = gt_labels[i] == gt_labels[j]
            pred_same = pred_labels[i] == pred_labels[j]
            if gt_same and pred_same:
                tp += 1
            elif pred_same and not gt_same:
                fp += 1
            elif gt_same and not pred_same:
                fn += 1

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0  # no pairs to evaluate → perfect by convention
    return 2 * tp / denom
