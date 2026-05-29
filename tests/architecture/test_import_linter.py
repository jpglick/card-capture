"""Import Linter contracts run on every default pytest invocation.

The PR lane installs `[dev]` extras so `lint-imports` is available. If the
binary is missing the test fails loudly with installation instructions
rather than skipping silently.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest


_INSTALL_HINT = (
    "lint-imports binary not found on PATH. Install dev extras:\n"
    "    python3 -m pip install -e '.[dev]'\n"
    "If your user-site bin is not on PATH, add it:\n"
    "    export PATH=\"$(python3 -m site --user-base)/bin:$PATH\""
)


def test_import_contracts() -> None:
    if shutil.which("lint-imports") is None:
        pytest.fail(_INSTALL_HINT)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:.:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        ["lint-imports"], capture_output=True, text=True, check=False, env=env
    )
    if result.returncode != 0:
        pytest.fail(
            f"Import Linter contract violations (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
