from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .models import QualityScore


@dataclass(frozen=True)
class ScoredCandidate:
    detection_id: int
    timestamp_ms: int
    image_path: str
    score: QualityScore


class CandidateSelector:
    def __init__(self, group_gap_ms: int = 1000, max_candidates: int = 10):
        self.group_gap_ms = group_gap_ms
        self.max_candidates = max_candidates

    def select(self, candidates: Iterable[ScoredCandidate]) -> List[ScoredCandidate]:
        ordered = sorted(candidates, key=lambda candidate: candidate.timestamp_ms)
        groups: List[List[ScoredCandidate]] = []
        for candidate in ordered:
            if not groups or candidate.timestamp_ms - groups[-1][-1].timestamp_ms > self.group_gap_ms:
                groups.append([candidate])
            else:
                groups[-1].append(candidate)

        best_per_group = [
            max(group, key=lambda candidate: (candidate.score.total, -candidate.timestamp_ms))
            for group in groups
        ]
        return sorted(
            best_per_group,
            key=lambda candidate: candidate.score.total,
            reverse=True,
        )[: self.max_candidates]
