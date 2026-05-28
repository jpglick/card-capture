"""Import Linter contracts. Advisory in Phase 1, blocking in Phase 2."""
from __future__ import annotations

import os
import subprocess

import pytest


def test_import_contracts():
    try:
        result = subprocess.run(
            ["lint-imports"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            pytest.fail(f"Import Linter violations:\n{result.stdout}\n{result.stderr}")
    except FileNotFoundError:
        pytest.skip("lint-imports not found; skipping blocking check")
