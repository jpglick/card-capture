"""Robustness metrics for card capture pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from tests.regression.matcher import MatchedPair
from tests.regression.pipeline_runner import HarnessInstance
from tests.regression.truth import ExpectedCard


def _angle_to_side(angle: str) -> str:
    """Convert angle string to side ('F' or 'B')."""
    a = angle.strip().lower()
    if a in {"front", "f"}:
        return "F"
    if a in {"back", "b"}:
        return "B"
    return "?"


@dataclass(frozen=True)
class RobustnessMetrics:
    """Core metrics for measuring pipeline robustness."""
    matched: Tuple[MatchedPair, ...]
    unmatched_truth: Tuple[ExpectedCard, ...]
    phantom_instances: Tuple[HarnessInstance, ...]
    truth_cards: Sequence[ExpectedCard]

    def card_recall(self) -> float:
        """Card recall: matched_cards / total_truth_cards.

        Returns the fraction of ground truth cards that were successfully detected.
        """
        total_truth = len(self.truth_cards)
        if total_truth == 0:
            return 1.0

        # Count unique truth cards in matched pairs
        matched_card_ids = {pair.truth_card.card_id for pair in self.matched}
        matched_count = len(matched_card_ids)

        return matched_count / total_truth

    def card_precision(self) -> float:
        """Card precision: matched_cards / total_predictions.

        Returns the fraction of predicted card instances that matched a truth card.
        """
        total_predictions = len(self.matched) + len(self.phantom_instances)
        if total_predictions == 0:
            return 1.0

        return len(self.matched) / total_predictions

    def front_back_f1(self) -> float:
        """F1 score for front/back angle accuracy.

        Measures whether the pipeline correctly assigns Front vs Back angles
        to matched card instances. F1 = 2 * (precision * recall) / (precision + recall).
        """
        if len(self.matched) == 0:
            return 1.0

        # Count correct angle assignments
        correct = sum(
            1 for pair in self.matched
            if _angle_to_side(pair.instance.angle) == pair.side
        )
        total = len(self.matched)

        # Precision and recall are both correct/total for this metric
        # (we measure correctness of angle assignment among matched pairs)
        accuracy = correct / total

        # For F1 on a single classifier, if precision == recall == accuracy,
        # then F1 = accuracy
        return accuracy

    def multi_card_scene_survival(self) -> float:
        """Recall restricted to multi-card and rapid-swap scenes.

        Measures robustness in challenging multi-card scenarios where
        multiple cards are visible simultaneously or swap rapidly.
        """
        multi_card_truth = tuple(
            card for card in self.truth_cards
            if card.scene_type in {"multi_card", "rapid_swap"}
        )

        if len(multi_card_truth) == 0:
            return 1.0

        # Count matched cards in multi-card scenes
        matched_card_ids = {pair.truth_card.card_id for pair in self.matched}
        matched_in_scenes = sum(
            1 for card in multi_card_truth
            if card.card_id in matched_card_ids
        )

        return matched_in_scenes / len(multi_card_truth)

    def foil_survival(self) -> float:
        """Recall for foil/holo cards.

        Measures robustness against reflective cards (foil, holo) which
        present challenging lighting and detection scenarios.
        """
        foil_truth = tuple(
            card for card in self.truth_cards
            if card.foil_label in {"foil", "holo"}
        )

        if len(foil_truth) == 0:
            return 1.0

        # Count matched foil cards
        matched_card_ids = {pair.truth_card.card_id for pair in self.matched}
        matched_foil = sum(
            1 for card in foil_truth
            if card.card_id in matched_card_ids
        )

        return matched_foil / len(foil_truth)

    def compute_all(self) -> Dict[str, float]:
        """Compute all metrics and return as a dictionary.

        Returns:
            Dict with keys: card_recall, card_precision, front_back_f1,
            multi_card_survival, foil_survival
        """
        return {
            "card_recall": self.card_recall(),
            "card_precision": self.card_precision(),
            "front_back_f1": self.front_back_f1(),
            "multi_card_survival": self.multi_card_scene_survival(),
            "foil_survival": self.foil_survival(),
        }
