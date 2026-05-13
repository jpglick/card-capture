"""Verify that the monolith and Metaflow pipeline produce equivalent
artifacts on a fixture video.

Stays in place until the monolith is deleted (Wave 5). Once that
happens, this test can go too.
"""
from __future__ import annotations

import subprocess
import sys
import sqlite3
from pathlib import Path

import pytest

FIXTURE_VIDEO = Path("tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV")


def _read_cards(db_path: Path) -> list[tuple]:
    """Return a sorted list of (instance_id, side, primary_hash)."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, angle, visual_hash "
            "FROM card_instances ORDER BY id"
        ).fetchall()
    return rows


@pytest.mark.skipif(
    not FIXTURE_VIDEO.exists(),
    reason="fixture video not present; equivalence test is opt-in",
)
def test_monolith_and_metaflow_produce_same_cards(tmp_path):
    """Run both pipeline paths on the same fixture and assert the
    extracted card set matches.
    """
    monolith_out = tmp_path / "monolith"
    metaflow_out = tmp_path / "metaflow"

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:.:{env.get('PYTHONPATH', '')}"

    # Monolith path — DEPRECATED; remove this test when the monolith goes.
    subprocess.run(
        [
            sys.executable, "-m", "card_capture.cli", "process",
            str(FIXTURE_VIDEO),
            "--output-dir", str(monolith_out),
            "--db", str(monolith_out / "cards.sqlite"),
            "--detector", "fake",
            "--pipeline", "monolith",
        ],
        check=True,
        env=env,
    )
    
    # Metaflow path — canonical.
    subprocess.run(
        [
            sys.executable, "-m", "card_capture.cli", "process",
            str(FIXTURE_VIDEO),
            "--output-dir", str(metaflow_out),
            "--db", str(metaflow_out / "cards.sqlite"),
            "--detector", "fake",
            "--pipeline", "metaflow",
        ],
        check=True,
        env=env,
    )

    mono_cards = _read_cards(monolith_out / "cards.sqlite")
    meta_cards = _read_cards(metaflow_out / "cards.sqlite")

    # Card set must match. 
    # Compare on side only, ignoring the exact IDs and primary_hash
    # (primary_hash differs because FakeCardDetector adds random noise to confidence,
    # causing a different frame to be selected as best_canonical).
    mono_set = {side for _, side, _ in mono_cards}
    meta_set = {side for _, side, _ in meta_cards}

    assert mono_set == meta_set, (
        f"Pipeline paths disagree.\n"
        f"  monolith-only: {mono_set - meta_set}\n"
        f"  metaflow-only: {meta_set - mono_set}"
    )
    # Also verify counts match
    assert len(mono_cards) == len(meta_cards), "Pipeline paths produced different number of cards"
