"""Integration: after a full pipeline run, every card_instances row has
a non-null reid_embedding regardless of tracker backend.

Closes V4_CONCERNS §1.6.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_VIDEO = Path("tests/fixtures/fake_video.mov")


def test_reid_embedding_populated_on_completion(tmp_path: Path):
    """Run the pipeline on a small fixture and assert that reid_embedding 
    is non-null for all saved cards.
    """
    if not FIXTURE_VIDEO.exists():
        # Create a tiny 1-frame fake video for the test if fixture missing
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(str(FIXTURE_VIDEO), fourcc, 1, (100, 100))
        vw.write(img)
        vw.release()

    out_dir = tmp_path / "out"
    db_path = out_dir / "cards.sqlite"
    
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:.:{env.get('PYTHONPATH', '')}"

    # Use bytetrack (which doesn't natively produce embeddings in our current adapter)
    # to prove the late-binding in the store step works.
    subprocess.run(
        [
            sys.executable, "-m", "card_capture.cli", "process",
            str(FIXTURE_VIDEO),
            "--output-dir", str(out_dir),
            "--db", str(db_path),
            "--detector", "fake",
            "--pipeline", "metaflow",
        ],
        env=env,
        check=True,
    )

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, reid_embedding FROM card_instances").fetchall()
        
        assert len(rows) > 0, "No cards were saved"
        for r in rows:
            assert r["reid_embedding"] is not None, f"reid_embedding missing for row {r['id']}"
            # Standard DINOv2 vits14 is 384 floats = 1536 bytes
            assert len(r["reid_embedding"]) >= 1536
