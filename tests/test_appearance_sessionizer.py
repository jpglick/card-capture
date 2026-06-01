from __future__ import annotations

import numpy as np

from card_capture.stages.track.appearance_sessionizer import (
    AppearanceObservation,
    AppearanceSessionizer,
)


def _unit(*values: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)


CARD_A = _unit(1.0, 0.0, 0.0, 0.0)
CARD_B = _unit(0.0, 1.0, 0.0, 0.0)
NOISE = _unit(0.0, 0.0, 1.0, 0.0)


def _obs(frame: int, emb: np.ndarray, novelty: float = 0.12) -> AppearanceObservation:
    return AppearanceObservation(
        frame_index=frame,
        detection_id=frame,
        embedding=emb,
        novelty_score=novelty,
    )


def test_direct_front_to_front_replacement_forms_two_plateaus():
    result = AppearanceSessionizer(confirm_frames=3).sessionize([
        _obs(0, CARD_A), _obs(1, CARD_A), _obs(2, CARD_A),
        _obs(3, CARD_B), _obs(4, CARD_B), _obs(5, CARD_B),
    ])
    assert [p.frame_range for p in result.retained_plateaus] == [(0, 2), (3, 5)]
    assert result.boundary_frame_indices == [3]


def test_isolated_transition_noise_does_not_create_plateau():
    result = AppearanceSessionizer(confirm_frames=3).sessionize([
        _obs(0, CARD_A), _obs(1, CARD_A), _obs(2, CARD_A),
        _obs(3, NOISE),
        _obs(4, CARD_B), _obs(5, CARD_B), _obs(6, CARD_B),
    ])
    assert [p.frame_range for p in result.retained_plateaus] == [(0, 2), (4, 6)]
    assert result.raw_jump_count == 2


def test_unconfirmed_tail_is_not_emitted_as_physical_card():
    result = AppearanceSessionizer(confirm_frames=3).sessionize([
        _obs(0, CARD_A), _obs(1, CARD_A), _obs(2, CARD_A),
        _obs(3, CARD_B), _obs(4, CARD_B),
    ])
    assert [p.frame_range for p in result.retained_plateaus] == [(0, 2)]


HOLDER = _unit(0.0, 0.0, 0.0, 1.0)


def _plateau(start: int, emb: np.ndarray, novelty: float, length: int = 3):
    return [_obs(start + offset, emb, novelty) for offset in range(length)]


def test_recurrent_short_high_novelty_bridge_is_suppressed():
    observations = (
        _plateau(0, CARD_A, 0.12, 5)
        + _plateau(10, HOLDER, 0.26, 3)
        + _plateau(20, CARD_B, 0.13, 5)
        + _plateau(30, HOLDER, 0.27, 3)
        + _plateau(40, NOISE, 0.14, 5)
        + _plateau(50, HOLDER, 0.25, 3)
        + _plateau(60, CARD_A, 0.11, 5)
    )
    result = AppearanceSessionizer(confirm_frames=3).sessionize(observations)
    assert len(result.suppressed_plateaus) == 3
    assert [p.frame_range for p in result.retained_plateaus] == [
        (0, 4), (20, 24), (40, 44), (60, 64),
    ]


def test_repeated_visual_duplicate_cards_remain_distinct_physical_sessions():
    observations = (
        _plateau(0, CARD_A, 0.12, 5)
        + _plateau(10, CARD_B, 0.13, 5)
        + _plateau(20, CARD_A, 0.12, 5)
    )
    result = AppearanceSessionizer(confirm_frames=3).sessionize(observations)
    assert [p.frame_range for p in result.retained_plateaus] == [
        (0, 4), (10, 14), (20, 24),
    ]


def test_recurrence_without_bridge_support_is_retained():
    observations = (
        _plateau(0, CARD_A, 0.12, 5)
        + _plateau(10, CARD_B, 0.13, 5)
        + _plateau(20, CARD_A, 0.12, 5)
        + _plateau(30, NOISE, 0.14, 5)
        + _plateau(40, CARD_A, 0.12, 5)
    )
    result = AppearanceSessionizer(confirm_frames=3).sessionize(observations)
    assert len(result.suppressed_plateaus) == 0
    assert len(result.retained_plateaus) == 5


def test_ambiguous_band_observations_do_not_create_a_new_plateau():
    # 0.8/0.6 unit vector is 0.2 cosine-distant from CARD_A: inside the
    # ambiguous (same=0.15, change=0.30] band. Such transition frames must not
    # form a session boundary even if several occur in a row.
    ambiguous = _unit(0.8, 0.6, 0.0, 0.0)
    result = AppearanceSessionizer(
        same_threshold=0.15, change_threshold=0.30, confirm_frames=3
    ).sessionize([
        _obs(0, CARD_A), _obs(1, CARD_A), _obs(2, CARD_A),
        _obs(3, ambiguous), _obs(4, ambiguous), _obs(5, ambiguous),
    ])
    assert [p.frame_range for p in result.retained_plateaus] == [(0, 2)]
    assert result.boundary_frame_indices == []
