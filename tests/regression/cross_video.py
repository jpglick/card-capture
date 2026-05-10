from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .matcher import MatchedPair


@dataclass(frozen=True)
class DedupMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def compute_dedup_f1(matched_pairs_per_video: Sequence[Sequence[MatchedPair]]) -> DedupMetrics:
    """Compare pipeline duplicate links to truth physical_card_key groupings."""

    instance_to_key: dict = {}
    for video_pairs in matched_pairs_per_video:
        for pair in video_pairs:
            if pair.truth_card.physical_card_key:
                instance_to_key[pair.instance.instance_id] = pair.truth_card.physical_card_key

    truth_pairs: set = set()
    by_key: dict = {}
    for iid, key in instance_to_key.items():
        by_key.setdefault(key, []).append(iid)
    for ids in by_key.values():
        ids.sort()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                truth_pairs.add((ids[i], ids[j]))

    pipeline_pairs: set = set()
    for video_pairs in matched_pairs_per_video:
        for pair in video_pairs:
            inst = pair.instance
            if inst.duplicate_of is None:
                continue
            a, b = sorted([inst.instance_id, inst.duplicate_of])
            pipeline_pairs.add((a, b))

    tp = len(pipeline_pairs & truth_pairs)
    fp = len(pipeline_pairs - truth_pairs)
    fn = len(truth_pairs - pipeline_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 1.0 if not truth_pairs else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return DedupMetrics(
        true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=precision, recall=recall, f1=f1,
    )
