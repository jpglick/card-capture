from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return 1.0 - float(np.dot(left, right))


def _normalized_centroid(observations: list["AppearanceObservation"]) -> np.ndarray:
    centroid = np.mean(np.stack([o.embedding for o in observations]), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 1e-12:
        raise ValueError("appearance centroid has zero norm")
    return np.asarray(centroid / norm, dtype=np.float32)


@dataclass(frozen=True)
class AppearanceObservation:
    frame_index: int
    detection_id: Hashable
    embedding: np.ndarray
    novelty_score: float


@dataclass
class AppearancePlateau:
    observations: list[AppearanceObservation]
    representative: np.ndarray
    suppressed_reason: str | None = None

    @property
    def frame_range(self) -> tuple[int, int]:
        return self.observations[0].frame_index, self.observations[-1].frame_index

    @property
    def median_novelty(self) -> float:
        return float(np.median([o.novelty_score for o in self.observations]))


@dataclass(frozen=True)
class AppearanceSessionizationResult:
    retained_plateaus: list[AppearancePlateau]
    suppressed_plateaus: list[AppearancePlateau]
    raw_jump_count: int
    boundary_frame_indices: list[int]
    frame_to_session_id: dict[int, int]

    def metrics(self) -> dict[str, object]:
        return {
            "appearance_raw_jumps": self.raw_jump_count,
            "appearance_plateaus_confirmed": len(self.retained_plateaus) + len(self.suppressed_plateaus),
            "appearance_bridge_plateaus_suppressed": len(self.suppressed_plateaus),
            "appearance_presentations_retained": len(self.retained_plateaus),
            "appearance_boundary_frames": list(self.boundary_frame_indices),
        }


@dataclass
class AppearanceSessionizer:
    same_threshold: float = 0.15
    change_threshold: float = 0.30
    confirm_frames: int = 3
    bridge_min_occurrences: int = 3
    bridge_position_ratio: float = 0.80
    bridge_neighbor_change_ratio: float = 0.80
    bridge_novelty_margin: float = 0.05
    bridge_max_length_ratio: float = 0.75

    def sessionize(
        self,
        observations: list[AppearanceObservation],
    ) -> AppearanceSessionizationResult:
        stable = self._form_stable_plateaus(observations)
        retained, suppressed = self._suppress_bridge_plateaus(stable)
        frame_to_session_id = {
            observation.frame_index: session_id
            for session_id, plateau in enumerate(retained)
            for observation in plateau.observations
        }
        return AppearanceSessionizationResult(
            retained_plateaus=retained,
            suppressed_plateaus=suppressed,
            raw_jump_count=self._raw_jump_count(observations),
            boundary_frame_indices=[p.frame_range[0] for p in retained[1:]],
            frame_to_session_id=frame_to_session_id,
        )

    def _raw_jump_count(self, observations: list[AppearanceObservation]) -> int:
        if not observations:
            return 0
        jumps = 0
        for i in range(len(observations) - 1):
            if cosine_distance(observations[i].embedding, observations[i+1].embedding) > self.change_threshold:
                jumps += 1
        return jumps

    def _form_stable_plateaus(self, observations: list[AppearanceObservation]) -> list[AppearancePlateau]:
        if not observations:
            return []
        
        plateaus = []
        active_buffer: list[AppearanceObservation] = []
        pending_buffer: list[AppearanceObservation] = []

        for obs in observations:
            if not active_buffer:
                active_buffer.append(obs)
                continue

            active_centroid = _normalized_centroid(active_buffer)
            dist_to_active = cosine_distance(obs.embedding, active_centroid)

            if dist_to_active <= self.same_threshold:
                active_buffer.append(obs)
                pending_buffer = []
            else:
                if dist_to_active <= self.change_threshold:
                    # Ambiguous transition noise: not close enough to extend the
                    # active plateau, not far enough to be a real replacement.
                    # Skip it without disturbing active/pending so a stray frame
                    # can never create a session boundary.
                    continue
                if not pending_buffer:
                    pending_buffer.append(obs)
                else:
                    pending_centroid = _normalized_centroid(pending_buffer)
                    dist_to_pending = cosine_distance(obs.embedding, pending_centroid)
                    if dist_to_pending <= self.same_threshold:
                        pending_buffer.append(obs)
                    else:
                        pending_buffer = [obs]

                if len(pending_buffer) >= self.confirm_frames:
                    if len(active_buffer) >= self.confirm_frames:
                        plateaus.append(AppearancePlateau(active_buffer, _normalized_centroid(active_buffer)))
                    active_buffer = pending_buffer
                    pending_buffer = []

        if len(active_buffer) >= self.confirm_frames:
            plateaus.append(AppearancePlateau(active_buffer, _normalized_centroid(active_buffer)))

        return plateaus

    def _suppress_bridge_plateaus(
        self,
        plateaus: list[AppearancePlateau],
    ) -> tuple[list[AppearancePlateau], list[AppearancePlateau]]:
        if not plateaus:
            return [], []

        # 1. Greedily cluster plateau representatives
        cluster_ids: list[int] = []
        cluster_representatives: list[np.ndarray] = []

        for p in plateaus:
            found_cluster = -1
            for cid, rep in enumerate(cluster_representatives):
                if cosine_distance(p.representative, rep) <= self.same_threshold:
                    found_cluster = cid
                    break
            if found_cluster == -1:
                found_cluster = len(cluster_representatives)
                cluster_representatives.append(p.representative)
            cluster_ids.append(found_cluster)

        # 2-6. Evaluate each cluster for bridge suppression
        suppressed_indices: set[int] = set()
        for cid in range(len(cluster_representatives)):
            occurrences = [i for i, cluster_id in enumerate(cluster_ids) if cluster_id == cid]
            if len(occurrences) < self.bridge_min_occurrences:
                continue

            # 3. Require interior_count / occurrence_count >= bridge_position_ratio
            interior_occurrences = [i for i in occurrences if 0 < i < len(plateaus) - 1]
            if len(interior_occurrences) / len(occurrences) < self.bridge_position_ratio:
                continue

            # 4. Require distinct neighbor ID fraction >= bridge_neighbor_change_ratio
            distinct_neighbor_count = 0
            for i in interior_occurrences:
                if cluster_ids[i-1] != cluster_ids[i+1]:
                    distinct_neighbor_count += 1
            
            if distinct_neighbor_count / len(interior_occurrences) < self.bridge_neighbor_change_ratio:
                continue

            # 5. Compute medians for candidate cluster and its neighbors
            candidate_lengths = [len(plateaus[i].observations) for i in occurrences]
            candidate_novelties = [plateaus[i].median_novelty for i in occurrences]

            neighbor_indices = []
            for i in occurrences:
                if i > 0:
                    neighbor_indices.append(i - 1)
                if i < len(plateaus) - 1:
                    neighbor_indices.append(i + 1)

            if not neighbor_indices:
                continue

            neighbor_lengths = [len(plateaus[i].observations) for i in neighbor_indices]
            neighbor_novelties = [plateaus[i].median_novelty for i in neighbor_indices]

            candidate_median_length = float(np.median(candidate_lengths))
            candidate_median_novelty = float(np.median(candidate_novelties))
            neighbor_median_length = float(np.median(neighbor_lengths))
            neighbor_median_novelty = float(np.median(neighbor_novelties))

            # 6. Suppress only if novelty margin or length ratio criteria are met
            novelty_pass = candidate_median_novelty >= (neighbor_median_novelty + self.bridge_novelty_margin)
            length_pass = candidate_median_length <= (neighbor_median_length * self.bridge_max_length_ratio)

            if novelty_pass or length_pass:
                for i in interior_occurrences:
                    suppressed_indices.add(i)

        retained: list[AppearancePlateau] = []
        suppressed: list[AppearancePlateau] = []
        for i, p in enumerate(plateaus):
            if i in suppressed_indices:
                p.suppressed_reason = "recurrent_bridge"
                suppressed.append(p)
            else:
                retained.append(p)

        return retained, suppressed
