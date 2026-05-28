# tests/pipeline/test_flow_runs.py
import subprocess
from pathlib import Path

FIXTURE_VIDEO = Path("tests/fixtures/golden_corpus/IMG_5872/IMG_5872.MOV")

def test_flow_runs_to_completion_on_fixture(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    db = tmp_path / "cards.sqlite"
    
    # We need to make sure the pipeline and card_capture packages are importable
    import sys
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:.:{env.get('PYTHONPATH', '')}"
    
    result = subprocess.run(
        [
            sys.executable, "-m", "pipeline.card_capture_flow", "run",
            "--video", str(FIXTURE_VIDEO),
            "--output-dir", str(out),
            "--db", str(db),
            "--detector", "fake",
        ],
        capture_output=True, text=True, timeout=120,
        env=env
    )
    
    print(result.stdout)
    print(result.stderr)
    
    assert result.returncode == 0, result.stderr
    assert (out / "frames").exists()
    assert db.exists()
