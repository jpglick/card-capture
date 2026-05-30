"""Synthetic test fixtures for the v5.5 back-half e2e tests.

We synthesise a tiny 480p MOV where two static rectangles ("cards")
appear in front of a checkerboard background. Deterministic via a
fixed seed; cached on disk between test runs.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


_CACHE = Path(__file__).parent / "fixtures"
_CACHE.mkdir(exist_ok=True)


def _make_checkerboard(h: int, w: int, square: int = 60) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, square):
        for x in range(0, w, square):
            if ((x // square) + (y // square)) % 2 == 0:
                img[y:y + square, x:x + square] = (180, 180, 180)
            else:
                img[y:y + square, x:x + square] = (60, 60, 60)
    return img


def _make_card(color, label, h=300, w=210) -> np.ndarray:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.putText(img, label, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 255, 255), 3)
    # Sprinkle texture so QualityScorer's sharpness component fires
    rng = np.random.RandomState(42)
    noise = rng.randint(-15, 15, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture(scope="session")
def synthetic_two_cards_mov() -> Path:
    """A 4-second 480x640 MOV with two cards held in succession."""
    out = _CACHE / "synthetic_two_cards.mov"
    if out.exists():
        return out

    w, h, fps, secs = 640, 480, 30, 4
    n_frames = fps * secs
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter could not open mp4v encoder on this platform")

    bg = _make_checkerboard(h, w)
    card_a = _make_card((40, 80, 200), "A")    # red-ish
    card_b = _make_card((200, 120, 40), "B")   # blue-ish

    for i in range(n_frames):
        frame = bg.copy()
        # First half: card A near top-left; second half: card B near center
        if i < n_frames // 2:
            card, x, y = card_a, 120, 80
        else:
            card, x, y = card_b, 220, 100
        frame[y:y + card.shape[0], x:x + card.shape[1]] = card
        writer.write(frame)

    writer.release()
    return out
